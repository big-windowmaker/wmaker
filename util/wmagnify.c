/*
 * magnify - a X utility for magnifying screen image
 *
 * 2000/5/21  Alfredo K. Kojima
 *
 * This program is in the Public Domain.
 *
 * 2026: faster rendering (MIT-SHM capture into reused buffers, scaling on
 * the client side and one image upload per frame instead of one rectangle
 * per magnified pixel), mouse wheel zoom with fractional factors, smooth
 * (bilinear) or pixel-perfect scaling, free resizing and an optional
 * always-on-top state. Big WindowMaker.
 */
#include <config.h>

#include <X11/Xproto.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>

#ifdef USE_XSHM
#include <sys/ipc.h>
#include <sys/shm.h>
#include <X11/extensions/XShm.h>
#endif

#include <WINGs/WINGs.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* 30 frames per second; a frame whose source pixels, position and zoom did
 * not change is not drawn again. */
int refreshrate = 33;

#define MIN_ZOOM 1.0
#define MAX_ZOOM 16.0
#define DEFAULT_ZOOM 2.0
#define MIN_WIDTH 120
#define MIN_HEIGHT 80

/* zoom steps of the mouse wheel and of +/- */
static const double zoomSteps[] = {1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 14, 16};
#define N_STEPS (sizeof(zoomSteps) / sizeof(zoomSteps[0]))

/* an image, in shared memory when the server allows it */
typedef struct {
	XImage *image;
	Display *dpy;
#ifdef USE_XSHM
	XShmSegmentInfo shm;
#endif
	Bool shared;
} Buffer;

typedef struct {
	WMWindow *win;
	WMFrame *frame;
	Window xwin;
	GC gc;

	int width, height;	/* size of the view, in screen pixels */
	double zoom;
	int refreshrate;

	Buffer src;		/* the captured area of the screen */
	Buffer dst;		/* what is shown, already magnified */
	uint32_t *last;		/* previous capture, to skip unchanged frames */
	size_t lastSize;
	int lastX, lastY;
	Bool dirty;		/* size, zoom or mode changed: draw */

	int x, y;		/* frozen position */
	Bool frozen;
	Bool smooth;
	Bool above;
	Bool markPointerHotspot;

	WMHandlerID tid;

	/* WMAGNIFY_STATS=1: frames per second on stderr */
	Bool stats;
	unsigned frames, captures;
	double statStart;
	double tCapture, tScale, tPut;	/* seconds spent, for the statistics */
} BufferData;

static BufferData *newWindow(double zoom);

int windowCount = 0;

Display *dpy, *vdpy;
WMScreen *scr;
unsigned int black;
WMColor *cursorColor1;
WMColor *cursorColor2;

static double now(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* ---- buffers ------------------------------------------------------------ */

#ifdef USE_XSHM
static Bool shmFailed;

static int shmErrorHandler(Display *d, XErrorEvent *e)
{
	(void)d;
	(void)e;
	shmFailed = True;
	return 0;
}
#endif

static void freeBuffer(Buffer *b)
{
	if (!b->image)
		return;
#ifdef USE_XSHM
	if (b->shared) {
		XShmDetach(b->dpy, &b->shm);
		XSync(b->dpy, False);
		b->image->data = NULL;
		XDestroyImage(b->image);
		shmdt(b->shm.shmaddr);
		b->image = NULL;
		b->shared = False;
		return;
	}
#endif
	XDestroyImage(b->image);
	b->image = NULL;
}

static Bool allocBuffer(Buffer *b, Display *d, int width, int height)
{
	int screen = DefaultScreen(d);
	Visual *visual = DefaultVisual(d, screen);
	int depth = DefaultDepth(d, screen);

	freeBuffer(b);
	b->dpy = d;
	b->shared = False;
#ifdef USE_XSHM
	if (XShmQueryExtension(d)) {
		b->image = XShmCreateImage(d, visual, depth, ZPixmap, NULL, &b->shm, width, height);
		if (b->image) {
			b->shm.shmid = shmget(IPC_PRIVATE, b->image->bytes_per_line * b->image->height,
					      IPC_CREAT | 0600);
			if (b->shm.shmid >= 0) {
				b->shm.shmaddr = b->image->data = shmat(b->shm.shmid, NULL, 0);
				b->shm.readOnly = False;
				if (b->shm.shmaddr != (void *)-1) {
					XErrorHandler old;

					shmFailed = False;
					old = XSetErrorHandler(shmErrorHandler);
					XShmAttach(d, &b->shm);
					XSync(d, False);
					XSetErrorHandler(old);
					/* removed now, freed when both sides detach */
					shmctl(b->shm.shmid, IPC_RMID, NULL);
					if (!shmFailed) {
						b->shared = True;
						return True;
					}
					shmdt(b->shm.shmaddr);
				} else {
					shmctl(b->shm.shmid, IPC_RMID, NULL);
				}
			}
			b->image->data = NULL;
			XDestroyImage(b->image);
			b->image = NULL;
		}
	}
#endif
	b->image = XCreateImage(d, visual, depth, ZPixmap, 0, NULL, width, height, 32, 0);
	if (!b->image)
		return False;
	b->image->data = malloc(b->image->bytes_per_line * height);
	if (!b->image->data) {
		XDestroyImage(b->image);
		b->image = NULL;
		return False;
	}
	return True;
}

/* ---- capture and scaling -------------------------------------------------- */

static Bool fast32(XImage *image)
{
	return image->bits_per_pixel == 32 && image->byte_order == LSBFirst;
}

static inline uint32_t srcPixel(XImage *image, int x, int y)
{
	if (fast32(image))
		return ((uint32_t *)(image->data + y * image->bytes_per_line))[x];
	return (uint32_t)XGetPixel(image, x, y);
}

static inline void dstPut(XImage *image, int x, int y, uint32_t p)
{
	if (fast32(image))
		((uint32_t *)(image->data + y * image->bytes_per_line))[x] = p;
	else
		XPutPixel(image, x, y, p);
}

/* bilinear mix of four 0x00RRGGBB pixels, weights in 1/256 */
static inline uint32_t mix(uint32_t a, uint32_t b, uint32_t c, uint32_t d, unsigned fx, unsigned fy)
{
	unsigned w00 = (256 - fx) * (256 - fy), w10 = fx * (256 - fy), w01 = (256 - fx) * fy, w11 = fx * fy;
	uint32_t r = (((a >> 16) & 0xff) * w00 + ((b >> 16) & 0xff) * w10 + ((c >> 16) & 0xff) * w01 +
		      ((d >> 16) & 0xff) * w11) >> 16;
	uint32_t g = (((a >> 8) & 0xff) * w00 + ((b >> 8) & 0xff) * w10 + ((c >> 8) & 0xff) * w01 +
		      ((d >> 8) & 0xff) * w11) >> 16;
	uint32_t bl = ((a & 0xff) * w00 + (b & 0xff) * w10 + (c & 0xff) * w01 + (d & 0xff) * w11) >> 16;
	return (r << 16) | (g << 8) | bl;
}

static void scaleInto(BufferData *data, int gx, int gy, int cx, int cy)
{
	XImage *src = data->src.image, *dst = data->dst.image;
	int sw = src->width, sh = src->height;
	double zoom = data->zoom;
	/* pointer position inside the captured image (the center of the view) */
	double ccx = cx - gx + 0.5, ccy = cy - gy + 0.5;
	int w = data->width, h = data->height;
	int *mapX = malloc(sizeof(int) * w);
	unsigned *fracX = malloc(sizeof(unsigned) * w);
	int *lo = malloc(sizeof(int) * w), *hi = malloc(sizeof(int) * w);
	int x, y;

	if (!mapX || !fracX || !lo || !hi) {
		free(mapX);
		free(fracX);
		free(lo);
		free(hi);
		return;
	}
	for (x = 0; x < w; x++) {
		double sx = ccx + (x + 0.5 - w / 2.0) / zoom;

		if (data->smooth) {
			sx -= 0.5;
			mapX[x] = (int)floor(sx);
			fracX[x] = (unsigned)((sx - floor(sx)) * 256);
		} else {
			mapX[x] = (int)floor(sx);
			fracX[x] = 0;
		}
		/* clamped neighbours for the smooth mode */
		lo[x] = mapX[x] < 0 ? 0 : mapX[x] >= sw ? sw - 1 : mapX[x];
		hi[x] = mapX[x] + 1 < 0 ? 0 : mapX[x] + 1 >= sw ? sw - 1 : mapX[x] + 1;
	}
	for (y = 0; y < h; y++) {
		double sy = ccy + (y + 0.5 - h / 2.0) / zoom;
		int iy;
		unsigned fy = 0;

		if (data->smooth) {
			sy -= 0.5;
			iy = (int)floor(sy);
			fy = (unsigned)((sy - floor(sy)) * 256);
		} else {
			iy = (int)floor(sy);
		}
		if (!data->smooth) {
			if (iy < 0 || iy >= sh) {
				for (x = 0; x < w; x++)
					dstPut(dst, x, y, black);
				continue;
			}
			if (fast32(src) && fast32(dst)) {
				uint32_t *s = (uint32_t *)(src->data + iy * src->bytes_per_line);
				uint32_t *d = (uint32_t *)(dst->data + y * dst->bytes_per_line);

				for (x = 0; x < w; x++)
					d[x] = (mapX[x] >= 0 && mapX[x] < sw) ? s[mapX[x]] : black;
			} else {
				for (x = 0; x < w; x++)
					dstPut(dst, x, y, (mapX[x] >= 0 && mapX[x] < sw) ? srcPixel(src, mapX[x], iy) : black);
			}
		} else {
			int y0 = iy < 0 ? 0 : iy >= sh ? sh - 1 : iy;
			int y1 = iy + 1 < 0 ? 0 : iy + 1 >= sh ? sh - 1 : iy + 1;

			if (fast32(src) && fast32(dst)) {
				const uint32_t *r0 = (const uint32_t *)(src->data + y0 * src->bytes_per_line);
				const uint32_t *r1 = (const uint32_t *)(src->data + y1 * src->bytes_per_line);
				uint32_t *d = (uint32_t *)(dst->data + y * dst->bytes_per_line);
				unsigned wy1 = fy, wy0 = 256 - fy;

				for (x = 0; x < w; x++) {
					int x0 = lo[x], x1 = hi[x];
					unsigned fx = fracX[x];
					uint32_t a = r0[x0], b = r0[x1], c = r1[x0], e = r1[x1];
					/* two channels at a time: 0x00RR00BB and 0x0000GG00 */
					uint32_t top_rb = ((a & 0xff00ff) * (256 - fx) + (b & 0xff00ff) * fx) >> 8;
					uint32_t top_g = ((a & 0x00ff00) * (256 - fx) + (b & 0x00ff00) * fx) >> 8;
					uint32_t bot_rb = ((c & 0xff00ff) * (256 - fx) + (e & 0xff00ff) * fx) >> 8;
					uint32_t bot_g = ((c & 0x00ff00) * (256 - fx) + (e & 0x00ff00) * fx) >> 8;
					uint32_t rb = (((top_rb & 0xff00ff) * wy0 + (bot_rb & 0xff00ff) * wy1) >> 8) & 0xff00ff;
					uint32_t g = (((top_g & 0x00ff00) * wy0 + (bot_g & 0x00ff00) * wy1) >> 8) & 0x00ff00;

					d[x] = rb | g;
				}
			} else {
				for (x = 0; x < w; x++)
					dstPut(dst, x, y, mix(srcPixel(src, lo[x], y0), srcPixel(src, hi[x], y0),
							      srcPixel(src, lo[x], y1), srcPixel(src, hi[x], y1), fracX[x], fy));
			}
		}
	}
	free(lo);
	free(hi);
	free(mapX);
	free(fracX);
}

static void putFrame(BufferData *data)
{
	int w, h;

	if (!data->dst.image || !data->xwin)
		return;
	/* right after a resize the image still has the previous size */
	w = data->dst.image->width;
	h = data->dst.image->height;
#ifdef USE_XSHM
	if (data->dst.shared)
		XShmPutImage(dpy, data->xwin, data->gc, data->dst.image, 0, 0, 0, 0, w, h, False);
	else
#endif
		XPutImage(dpy, data->xwin, data->gc, data->dst.image, 0, 0, 0, 0, w, h);

	if (data->markPointerHotspot && !data->frozen) {
		XRectangle rects[4];
		int m = data->zoom < 2 ? 2 : (int)data->zoom;
		int cx = data->width / 2, cy = data->height / 2;

		rects[0].x = cx - 3 * m;
		rects[0].y = cy;
		rects[0].width = 2 * m;
		rects[0].height = m;
		rects[1].x = cx + 2 * m;
		rects[1].y = cy;
		rects[1].width = 2 * m;
		rects[1].height = m;
		XFillRectangles(dpy, data->xwin, WMColorGC(cursorColor1), rects, 2);
		rects[2].x = cx;
		rects[2].y = cy - 3 * m;
		rects[2].width = m;
		rects[2].height = 2 * m;
		rects[3].x = cx;
		rects[3].y = cy + 2 * m;
		rects[3].width = m;
		rects[3].height = 2 * m;
		XFillRectangles(dpy, data->xwin, WMColorGC(cursorColor2), rects + 2, 2);
	}
}

static void updateImage(BufferData *data, int cx, int cy)
{
	int scrW = WidthOfScreen(DefaultScreenOfDisplay(vdpy));
	int scrH = HeightOfScreen(DefaultScreenOfDisplay(vdpy));
	int sw = (int)ceil(data->width / data->zoom) + 2;
	int sh = (int)ceil(data->height / data->zoom) + 2;
	int gx, gy;
	size_t bytes;

	if (sw > scrW)
		sw = scrW;
	if (sh > scrH)
		sh = scrH;
	if (!data->src.image || data->src.image->width != sw || data->src.image->height != sh) {
		if (!allocBuffer(&data->src, vdpy, sw, sh))
			return;
		data->dirty = True;
	}
	if (!data->dst.image || data->dst.image->width != data->width || data->dst.image->height != data->height) {
		if (!allocBuffer(&data->dst, dpy, data->width, data->height))
			return;
		data->dirty = True;
	}

	/* the captured area stays inside the screen: near an edge the pointer
	 * is shown off center instead of a black border */
	gx = cx - sw / 2;
	gy = cy - sh / 2;
	if (gx < 0)
		gx = 0;
	if (gy < 0)
		gy = 0;
	if (gx + sw > scrW)
		gx = scrW - sw;
	if (gy + sh > scrH)
		gy = scrH - sh;

	double t0 = data->stats ? now() : 0;
#ifdef USE_XSHM
	if (data->src.shared) {
		if (!XShmGetImage(vdpy, DefaultRootWindow(vdpy), data->src.image, gx, gy, AllPlanes))
			return;
	} else
#endif
	{
		if (!XGetSubImage(vdpy, DefaultRootWindow(vdpy), gx, gy, sw, sh, AllPlanes, ZPixmap,
				  data->src.image, 0, 0))
			return;
	}
	data->captures++;
	if (data->stats)
		data->tCapture += now() - t0;

	bytes = (size_t)data->src.image->bytes_per_line * sh;
	if (!data->dirty && data->last && data->lastSize == bytes && data->lastX == cx && data->lastY == cy &&
	    memcmp(data->last, data->src.image->data, bytes) == 0)
		return;	/* nothing changed */
	if (data->lastSize != bytes) {
		free(data->last);
		data->last = malloc(bytes);
		data->lastSize = data->last ? bytes : 0;
	}
	if (data->last)
		memcpy(data->last, data->src.image->data, bytes);
	data->lastX = cx;
	data->lastY = cy;
	data->dirty = False;

	t0 = data->stats ? now() : 0;
	scaleInto(data, gx, gy, cx, cy);
	if (data->stats) {
		double t1 = now();

		data->tScale += t1 - t0;
		t0 = t1;
	}
	putFrame(data);
	if (data->stats) {
		XSync(dpy, False);
		data->tPut += now() - t0;
	}
	data->frames++;
}

static void updateTitle(BufferData *data)
{
	char buf[64], zoom[16];

	if (fabs(data->zoom - round(data->zoom)) < 0.001)
		snprintf(zoom, sizeof(zoom), "%d", (int)round(data->zoom));
	else
		snprintf(zoom, sizeof(zoom), "%.2g", data->zoom);
	snprintf(buf, sizeof(buf), data->frozen ? "[Magnify %sx%s]" : "Magnify %sx%s", zoom,
		 data->smooth ? " (smooth)" : " (pixels)");
	WMSetWindowTitle(data->win, buf);
}

static void update(void *d)
{
	BufferData *data = (BufferData *) d;
	Window win;
	int rx, ry;
	int bla;
	unsigned ubla;

	if (data->frozen) {
		rx = data->x;
		ry = data->y;
	} else {
		XQueryPointer(dpy, DefaultRootWindow(dpy), &win, &win, &rx, &ry, &bla, &bla, &ubla);
	}
	updateImage(data, rx, ry);

	if (data->stats) {
		double t = now();

		if (t - data->statStart >= 5) {
			fprintf(stderr, "wmagnify: %.1f frames/s drawn, %.1f captures/s, zoom %.2f, %dx%d, "
				"ms per frame: capture %.2f scale %.2f put %.2f\n",
				data->frames / (t - data->statStart), data->captures / (t - data->statStart),
				data->zoom, data->width, data->height,
				data->captures ? 1000 * data->tCapture / data->captures : 0,
				data->frames ? 1000 * data->tScale / data->frames : 0,
				data->frames ? 1000 * data->tPut / data->frames : 0);
			data->frames = data->captures = 0;
			data->tCapture = data->tScale = data->tPut = 0;
			data->statStart = t;
		}
	}

	data->tid = WMAddTimerHandler(data->refreshrate, update, data);
}

static void setZoom(BufferData *data, double zoom)
{
	if (zoom < MIN_ZOOM)
		zoom = MIN_ZOOM;
	if (zoom > MAX_ZOOM)
		zoom = MAX_ZOOM;
	if (zoom == data->zoom)
		return;
	data->zoom = zoom;
	data->dirty = True;
	updateTitle(data);
	update(data);
	WMDeleteTimerHandler(data->tid);
	data->tid = WMAddTimerHandler(data->refreshrate, update, data);
}

static void stepZoom(BufferData *data, int dir)
{
	size_t i;

	if (dir > 0) {
		for (i = 0; i < N_STEPS; i++)
			if (zoomSteps[i] > data->zoom + 0.001) {
				setZoom(data, zoomSteps[i]);
				return;
			}
	} else {
		for (i = N_STEPS; i-- > 0;)
			if (zoomSteps[i] < data->zoom - 0.001) {
				setZoom(data, zoomSteps[i]);
				return;
			}
	}
}

/* _NET_WM_STATE_ABOVE: kept above the other windows */
static void setAbove(BufferData *data, Bool above)
{
	XEvent ev;

	data->above = above;
	memset(&ev, 0, sizeof(ev));
	ev.xclient.type = ClientMessage;
	ev.xclient.window = WMWidgetXID(data->win);
	ev.xclient.message_type = XInternAtom(dpy, "_NET_WM_STATE", False);
	ev.xclient.format = 32;
	ev.xclient.data.l[0] = above ? 1 : 0;
	ev.xclient.data.l[1] = XInternAtom(dpy, "_NET_WM_STATE_ABOVE", False);
	ev.xclient.data.l[3] = 1;
	XSendEvent(dpy, DefaultRootWindow(dpy), False, SubstructureRedirectMask | SubstructureNotifyMask, &ev);
	XFlush(dpy);
}

static void resizedWindow(void *d, WMNotification * notif)
{
	BufferData *data = (BufferData *) d;
	WMView *view = (WMView *) WMGetNotificationObject(notif);
	WMSize size;

	size = WMGetViewSize(view);
	if ((int)size.width == data->width && (int)size.height == data->height)
		return;
	data->width = size.width;
	data->height = size.height;
	WMResizeWidget(data->frame, size.width, size.height);
	data->dirty = True;
}

static void exposeHandler(XEvent *event, void *d)
{
	BufferData *data = (BufferData *) d;

	if (event->xexpose.count == 0)
		putFrame(data);
}

static void buttonHandler(XEvent *event, void *d)
{
	BufferData *data = (BufferData *) d;

	switch (event->xbutton.button) {
	case Button4:
		stepZoom(data, 1);
		break;
	case Button5:
		stepZoom(data, -1);
		break;
	case Button2:
		setZoom(data, DEFAULT_ZOOM);
		break;
	}
}

static void closeWindow(WMWidget * w, void *d)
{
	BufferData *data = (BufferData *) d;

	windowCount--;
	if (windowCount == 0) {
		WMReleaseApplication();
		exit(0);
	} else {
		WMDeleteTimerHandler(data->tid);
		WMDestroyWidget(w);
		freeBuffer(&data->src);
		freeBuffer(&data->dst);
		free(data->last);
		XFreeGC(dpy, data->gc);
		wfree(data);
	}
}

static void keyHandler(XEvent * event, void *d)
{
	BufferData *data = (BufferData *) d;
	char buf[32];
	KeySym ks;

	if (XLookupString(&event->xkey, buf, 31, &ks, NULL) > 0) {
		switch (buf[0]) {
		case 'n':
			newWindow(data->zoom);
			break;
		case 'm':
			data->markPointerHotspot = !data->markPointerHotspot;
			data->dirty = True;
			break;
		case 'f':
		case ' ':
			data->frozen = !data->frozen;
			if (data->frozen) {
				data->x = event->xkey.x_root;
				data->y = event->xkey.y_root;
			}
			data->dirty = True;
			updateTitle(data);
			break;
		case 's':
			data->smooth = !data->smooth;
			data->dirty = True;
			updateTitle(data);
			break;
		case 'a':
			setAbove(data, !data->above);
			break;
		case '+':
		case '=':
			stepZoom(data, 1);
			break;
		case '-':
			stepZoom(data, -1);
			break;
		case '0':
			setZoom(data, DEFAULT_ZOOM);
			break;
		case '1':
		case '2':
		case '3':
		case '4':
		case '5':
		case '6':
		case '7':
		case '8':
		case '9':
			setZoom(data, buf[0] - '0');
			break;
		}
	}
}

static Bool startAbove = True;
static Bool startSmooth = False;

static BufferData *newWindow(double zoom)
{
	WMWindow *win;
	WMFrame *frame;
	BufferData *data;

	windowCount++;

	data = wmalloc(sizeof(BufferData));
	data->zoom = zoom;
	data->refreshrate = refreshrate;
	data->smooth = startSmooth;
	data->width = 300;
	data->height = 200;
	data->dirty = True;
	data->stats = getenv("WMAGNIFY_STATS") != NULL;
	data->statStart = now();

	win = WMCreateWindow(scr, "magnify");
	data->win = win;
	WMResizeWidget(win, data->width, data->height);
	WMSetWindowMinSize(win, MIN_WIDTH, MIN_HEIGHT);
	WMSetViewNotifySizeChanges(WMWidgetView(win), True);
	updateTitle(data);

	frame = WMCreateFrame(win);
	data->frame = frame;
	WMSetFrameRelief(frame, WRFlat);
	WMResizeWidget(frame, data->width, data->height);
	WMMoveWidget(frame, 0, 0);

	WMCreateEventHandler(WMWidgetView(win), KeyReleaseMask, keyHandler, data);
	WMCreateEventHandler(WMWidgetView(frame), ButtonPressMask, buttonHandler, data);
	WMCreateEventHandler(WMWidgetView(frame), ExposureMask, exposeHandler, data);

	WMAddNotificationObserver(resizedWindow, data, WMViewSizeDidChangeNotification, WMWidgetView(win));

	WMRealizeWidget(win);

	WMMapSubwidgets(win);
	WMMapWidget(win);

	data->xwin = WMWidgetXID(frame);
	data->gc = XCreateGC(dpy, data->xwin, 0, NULL);
	/* the frame is fully covered by the image: no background flashes */
	XSetWindowBackgroundPixmap(dpy, data->xwin, None);

	WMSetWindowCloseAction(win, closeWindow, data);
	if (startAbove)
		setAbove(data, True);
	data->tid = WMAddTimerHandler(data->refreshrate, update, data);

	return data;
}

int main(int argc, char **argv)
{
	int i;
	char *display = "";
	char *vdisplay = NULL;
	double magfactor = DEFAULT_ZOOM;

	WMInitializeApplication("Magnify", &argc, argv);

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-display") == 0) {
			i++;
			if (i >= argc)
				goto help;
			display = argv[i];
		} else if (strcmp(argv[i], "-vdisplay") == 0) {
			i++;
			if (i >= argc)
				goto help;
			vdisplay = argv[i];
		} else if (strcmp(argv[i], "-m") == 0) {
			i++;
			if (i >= argc)
				goto help;
			magfactor = atof(argv[i]);
			if (magfactor < MIN_ZOOM || magfactor > MAX_ZOOM) {
				printf("%s:invalid magnification factor ``%s''\n", argv[0], argv[i]);
				exit(1);
			}
		} else if (strcmp(argv[i], "-r") == 0) {
			i++;
			if (i >= argc)
				goto help;
			refreshrate = atoi(argv[i]);
			if (refreshrate < 1) {
				printf("%s:invalid refresh rate ``%s''\n", argv[0], argv[i]);
				exit(1);
			}
		} else if (strcmp(argv[i], "-smooth") == 0) {
			startSmooth = True;
		} else if (strcmp(argv[i], "-noabove") == 0) {
			startAbove = False;
		} else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
 help:

			printf("Usage: %s [options]\n", argv[0]);
			puts("Options:");
			puts("  -display <display>\tdisplay where to magnification is shown");
			puts("  -m <number>\t\tchange magnification factor (1 to 16, default 2)");
			puts("  -r <number>\t\tchange refresh delay, in milliseconds (default 33)");
			puts("  -smooth\t\tsmooth (bilinear) scaling instead of pixel-perfect");
			puts("  -noabove\t\tdo not keep the window above the others");
			puts("  -vdisplay <display>\tdisplay from which the magnification is taken");
			puts("  -h, --help\t\tdisplay this help page");
			puts("Mouse:");
			puts("  wheel up/down		zoom in/out");
			puts("  middle button		back to 2x");
			puts("Keys:");
			puts("  1,2,3,4,5,6,7,8,9	change the magnification factor");
			puts("  +, -			zoom in/out in smaller steps");
			puts("  0			back to 2x");
			puts("  s			smooth or pixel-perfect scaling");
			puts("  a			keep above the other windows (on/off)");
			puts("  <space>, f		freeze the 'camera', making it magnify only the current\n"
			     "			position");
			puts("  n			create a new window");
			puts("  m			show/hide the pointer hotspot mark");
			exit(0);
		}
	}

	dpy = XOpenDisplay(display);
	if (!dpy) {
		puts("could not open display");
		exit(1);
	}

	if (vdisplay) {
		vdpy = XOpenDisplay(vdisplay);
		if (!vdpy) {
			puts("could not open display to be viewed");
			exit(1);
		}
	} else {
		vdpy = dpy;
	}

	black = BlackPixel(dpy, DefaultScreen(dpy));

	scr = WMCreateScreen(dpy, 0);

	cursorColor1 = WMCreateNamedColor(scr, "#ff0000", False);
	cursorColor2 = WMCreateNamedColor(scr, "#00ff00", False);

	newWindow(magfactor);

	WMScreenMainLoop(scr);

	return 0;
}
