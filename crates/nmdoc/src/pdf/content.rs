//! The content-stream interpreter for text: the graphics state stack (`q Q cm`), the text state
//! (`BT ET Tf Td TD Tm T* Tj TJ ' " Tc Tw Tz TL Ts Tr`), form XObjects (`Do` recurses), images and
//! inline images (counted by the page area they cover), and marked content (`BDC BMC EMC`, with
//! `MCID`, `Artifact` and `ActualText`). Every glyph comes out placed on the page as displayed
//! (rotation and crop box applied; y grows downward), with its font size, font and marked
//! content id.
use super::file::File;
use super::font::Font;
use super::lex::{Dict, Lexer, Obj};
use std::collections::HashMap;
use std::rc::Rc;

/// An affine matrix `[a b c d e f]` (PDF row-vector form).
pub type Mat = [f64; 6];

const ID: Mat = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// `a` then `b`.
#[must_use]
pub fn mul(a: &Mat, b: &Mat) -> Mat {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    ]
}

fn apply(m: &Mat, x: f64, y: f64) -> (f64, f64) {
    (x * m[0] + y * m[2] + m[4], x * m[1] + y * m[3] + m[5])
}

/// One placed glyph.
#[derive(Clone, Debug)]
pub struct Glyph {
    /// Its characters; empty when the font maps it to none.
    pub text: String,
    pub x0: f64,
    pub x1: f64,
    /// Baseline, page units from the top.
    pub y: f64,
    /// Font size on the page.
    pub size: f64,
    pub bold: bool,
    pub mono: bool,
    /// Innermost marked content id.
    pub mcid: Option<u32>,
    /// Inside an `Artifact` marked content sequence (running headers, page numbers).
    pub artifact: bool,
    /// No character map for this code.
    pub unmapped: bool,
    /// The baseline runs left to right (not rotated or vertical).
    pub upright: bool,
}

/// What one page holds.
#[derive(Clone, Debug, Default)]
pub struct Page {
    pub glyphs: Vec<Glyph>,
    pub width: f64,
    pub height: f64,
    /// Share of the page covered by images (sum of their areas over the page area).
    pub image_cover: f64,
    pub images: usize,
    /// Streams skipped, by filter name.
    pub skipped: Vec<&'static str>,
}

#[derive(Clone)]
struct State {
    ctm: Mat,
    font: Option<Rc<Font>>,
    size: f64,
    tc: f64,
    tw: f64,
    th: f64,
    tl: f64,
    rise: f64,
}

struct Marked {
    mcid: Option<u32>,
    artifact: bool,
    actual: Option<(String, usize)>,
}

/// Interpreter state over one page.
struct Run<'f, 'a> {
    file: &'f File<'a>,
    page: Page,
    /// Fonts by object number or by resource name with its resources' identity.
    fonts: HashMap<String, Rc<Font>>,
    marked: Vec<Marked>,
    disp: Mat,
    ops: usize,
}

/// Most operators run on one page (all forms included).
const MAX_OPS: usize = 5_000_000;
/// Deepest nesting of form XObjects.
const MAX_FORMS: usize = 12;

fn num(o: Option<&Obj>) -> f64 {
    o.and_then(Obj::as_num).filter(|x| x.is_finite()).unwrap_or(0.0)
}

fn rect(file: &File<'_>, d: &Dict, key: &str) -> Option<[f64; 4]> {
    let a = file.at(d, key);
    let a = a.as_array()?;
    let v: Vec<f64> = a.iter().filter_map(|x| file.resolve(x).as_num()).collect();
    let (&x0, &y0, &x1, &y1) = (v.first()?, v.get(1)?, v.get(2)?, v.get(3)?);
    Some([x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1)])
}

/// Read one page's text.
#[must_use]
pub fn page(file: &File<'_>, d: &Dict) -> Page {
    let media = rect(file, d, "MediaBox").unwrap_or([0.0, 0.0, 612.0, 792.0]);
    let crop = rect(file, d, "CropBox").map_or(media, |c| [c[0].max(media[0]), c[1].max(media[1]), c[2].min(media[2]), c[3].min(media[3])]);
    let crop = if crop[2] - crop[0] < 1.0 || crop[3] - crop[1] < 1.0 { media } else { crop };
    let rot = (file.at(d, "Rotate").as_int().unwrap_or(0) % 360 + 360) % 360;
    let [x0, y0, x1, y1] = crop;
    // From user space to the page as displayed, y downward.
    let (disp, w, h) = match rot {
        90 => ([0.0, 1.0, 1.0, 0.0, -y0, -x0], y1 - y0, x1 - x0),
        180 => ([-1.0, 0.0, 0.0, 1.0, x1, -y0], x1 - x0, y1 - y0),
        270 => ([0.0, -1.0, -1.0, 0.0, y1, x1], y1 - y0, x1 - x0),
        _ => ([1.0, 0.0, 0.0, -1.0, -x0, y1], x1 - x0, y1 - y0),
    };
    let mut run = Run { file, page: Page { width: w, height: h, ..Page::default() }, fonts: HashMap::new(), marked: Vec::new(), disp, ops: 0 };
    let res = file.at(d, "Resources");
    let res = res.as_dict().cloned().unwrap_or_default();
    let mut data = Vec::new();
    match file.at(d, "Contents") {
        Obj::Stream(s) => match file.stream_data(&s) {
            Ok(b) => data = b,
            Err(e) => run.skip(e),
        },
        Obj::Array(parts) => {
            for p in parts {
                if let Obj::Stream(s) = file.resolve(&p) {
                    match file.stream_data(&s) {
                        Ok(b) => {
                            data.extend_from_slice(&b);
                            data.push(b'\n');
                        }
                        Err(e) => run.skip(e),
                    }
                }
            }
        }
        _ => {}
    }
    let st = State { ctm: ID, font: None, size: 0.0, tc: 0.0, tw: 0.0, th: 1.0, tl: 0.0, rise: 0.0 };
    run.stream(&data, &res, st, 0, &mut Vec::new());
    run.page
}

impl Run<'_, '_> {
    fn skip(&mut self, e: super::filters::Skip) {
        let name = match e {
            super::filters::Skip::Image(n) => n,
            super::filters::Skip::Unknown => "unknown filter",
            super::filters::Skip::Broken => "damaged stream",
        };
        if !self.page.skipped.contains(&name) {
            self.page.skipped.push(name);
        }
    }

    fn font(&mut self, res: &Dict, name: &str) -> Rc<Font> {
        let fonts = self.file.at(res, "Font");
        let entry = fonts.as_dict().and_then(|f| f.get(name)).cloned();
        let key = match &entry {
            Some(Obj::Ref(n, _)) => format!("#{n}"),
            _ => format!("{name}@{:p}", std::ptr::from_ref(res)),
        };
        if let Some(f) = self.fonts.get(&key) {
            return f.clone();
        }
        let f = match entry.map(|e| self.file.resolve(&e)) {
            Some(o) => o.as_dict().map_or_else(Font::unknown, |d| Font::load(self.file, d)),
            None => Font::unknown(),
        };
        let f = Rc::new(f);
        self.fonts.insert(key, f.clone());
        f
    }

    fn cover(&mut self, ctm: &Mat) {
        let m = mul(ctm, &self.disp);
        let area = (m[0] * m[3] - m[1] * m[2]).abs();
        let page = (self.page.width * self.page.height).max(1.0);
        self.page.image_cover += (area / page).min(1.0);
        self.page.images += 1;
    }

    /// Show a string: one glyph per code, the text matrix moved past each.
    fn show(&mut self, s: &[u8], st: &State, tm: &mut Mat) {
        let Some(font) = st.font.clone() else { return };
        let (mcid, artifact) = self.marked.iter().rev().find_map(|m| m.mcid.map(|c| (Some(c), m.artifact))).unwrap_or((None, self.marked.iter().any(|m| m.artifact)));
        let artifact = artifact || self.marked.iter().any(|m| m.artifact);
        for (code, len) in font.codes(s) {
            let w0 = font.width(code);
            let trm = mul(&mul(&[st.size * st.th, 0.0, 0.0, st.size, 0.0, st.rise], tm), &mul(&st.ctm, &self.disp));
            let (ox, oy) = apply(&trm, 0.0, 0.0);
            let (ex, ey) = apply(&trm, w0, 0.0);
            let (ux, uy) = (trm[2], trm[3]);
            let size = (ux * ux + uy * uy).sqrt();
            let text = font.text(code);
            let is_space = len == 1 && code == 32;
            let (dx, dy) = (ex - ox, ey - oy);
            let upright = dx >= 0.0 && dy.abs() <= dx.abs() * 0.2 + 0.01;
            let inside = ox > -5.0 && ox < self.page.width + 5.0 && oy > -5.0 && oy < self.page.height + 5.0;
            if inside && size > 0.5 {
                self.page.glyphs.push(Glyph {
                    unmapped: text.is_none() && !is_space,
                    text: text.unwrap_or_else(|| if is_space { " ".to_owned() } else { String::new() }),
                    x0: ox.min(ex),
                    x1: ox.max(ex),
                    y: oy,
                    size,
                    bold: font.bold,
                    mono: font.mono,
                    mcid,
                    artifact,
                    upright,
                });
            }
            let tx = (w0 * st.size + st.tc + if is_space { st.tw } else { 0.0 }) * st.th;
            *tm = mul(&[1.0, 0.0, 0.0, 1.0, tx, 0.0], tm);
        }
    }

    fn mark_begin(&mut self, tag: &str, props: Option<&Obj>, res: &Dict) {
        let props = match props {
            Some(Obj::Name(n)) => self.file.at(res, "Properties").as_dict().and_then(|p| p.get(n)).map(|x| self.file.resolve(x)),
            Some(o) => Some(self.file.resolve(o)),
            None => None,
        };
        let d = props.as_ref().and_then(Obj::as_dict);
        let mcid = d.and_then(|d| d.int("MCID")).and_then(|x| u32::try_from(x).ok());
        let actual = d.and_then(|d| d.get("ActualText")).and_then(Obj::as_str).map(|s| (super::file::text_string(s), self.page.glyphs.len()));
        self.marked.push(Marked { mcid, artifact: tag == "Artifact", actual });
    }

    fn mark_end(&mut self) {
        let Some(m) = self.marked.pop() else { return };
        // ActualText replaces the glyphs it covers by one glyph spanning them.
        if let Some((text, from)) = m.actual
            && from < self.page.glyphs.len()
        {
            let part: Vec<Glyph> = self.page.glyphs.drain(from..).collect();
            if let Some(first) = part.first() {
                let mut g = first.clone();
                g.x1 = part.iter().map(|x| x.x1).fold(g.x1, f64::max);
                g.x0 = part.iter().map(|x| x.x0).fold(g.x0, f64::min);
                g.text = text;
                g.unmapped = false;
                self.page.glyphs.push(g);
            }
        }
    }

    #[allow(clippy::too_many_lines, reason = "one match over the operators")]
    fn stream(&mut self, data: &[u8], res: &Dict, mut st: State, depth: usize, forms: &mut Vec<u32>) {
        let mut stack: Vec<State> = Vec::new();
        let mut args: Vec<Obj> = Vec::new();
        let mut tm = ID;
        let mut tlm = ID;
        let mut lx = Lexer::new(data, 0);
        while let Some(t) = lx.object() {
            let op = match t {
                Ok(o) => {
                    if args.len() < 64 {
                        args.push(o);
                    }
                    continue;
                }
                Err(w) => w,
            };
            self.ops += 1;
            if self.ops > MAX_OPS {
                return;
            }
            let a = |i: usize| num(args.get(i));
            match op.as_slice() {
                b"q" => {
                    if stack.len() < 64 {
                        stack.push(st.clone());
                    }
                }
                b"Q" => {
                    if let Some(s) = stack.pop() {
                        st = s;
                    }
                }
                b"cm" if args.len() >= 6 => st.ctm = mul(&[a(0), a(1), a(2), a(3), a(4), a(5)], &st.ctm),
                b"BT" => {
                    tm = ID;
                    tlm = ID;
                }
                b"Tf" => {
                    if let Some(n) = args.first().and_then(Obj::as_name) {
                        st.font = Some(self.font(res, n));
                    }
                    st.size = a(1);
                }
                b"Tc" => st.tc = a(0),
                b"Tw" => st.tw = a(0),
                b"Tz" => st.th = a(0) / 100.0,
                b"TL" => st.tl = a(0),
                b"Ts" => st.rise = a(0),
                b"Td" => {
                    tlm = mul(&[1.0, 0.0, 0.0, 1.0, a(0), a(1)], &tlm);
                    tm = tlm;
                }
                b"TD" => {
                    st.tl = -a(1);
                    tlm = mul(&[1.0, 0.0, 0.0, 1.0, a(0), a(1)], &tlm);
                    tm = tlm;
                }
                b"Tm" if args.len() >= 6 => {
                    tlm = [a(0), a(1), a(2), a(3), a(4), a(5)];
                    tm = tlm;
                }
                b"T*" => {
                    tlm = mul(&[1.0, 0.0, 0.0, 1.0, 0.0, -st.tl], &tlm);
                    tm = tlm;
                }
                b"Tj" => {
                    if let Some(Obj::Str(s)) = args.first() {
                        let s = s.clone();
                        self.show(&s, &st, &mut tm);
                    }
                }
                b"'" | b"\"" => {
                    if op.as_slice() == b"\"" {
                        st.tw = a(0);
                        st.tc = a(1);
                    }
                    tlm = mul(&[1.0, 0.0, 0.0, 1.0, 0.0, -st.tl], &tlm);
                    tm = tlm;
                    if let Some(Obj::Str(s)) = args.last() {
                        let s = s.clone();
                        self.show(&s, &st, &mut tm);
                    }
                }
                b"TJ" => {
                    if let Some(Obj::Array(items)) = args.first() {
                        let items = items.clone();
                        for it in &items {
                            match it {
                                Obj::Str(s) => self.show(s, &st, &mut tm),
                                o => {
                                    let n = o.as_num().unwrap_or(0.0);
                                    tm = mul(&[1.0, 0.0, 0.0, 1.0, -n / 1000.0 * st.size * st.th, 0.0], &tm);
                                }
                            }
                        }
                    }
                }
                b"Do" => {
                    if let Some(n) = args.first().and_then(Obj::as_name) {
                        let entry = self.file.at(res, "XObject").as_dict().and_then(|x| x.get(n)).cloned();
                        self.xobject(entry, res, &st, depth, forms);
                    }
                }
                b"BI" => {
                    self.inline_image(&mut lx);
                    self.cover(&st.ctm);
                }
                b"BMC" => {
                    let tag = args.first().and_then(Obj::as_name).unwrap_or("").to_owned();
                    self.mark_begin(&tag, None, res);
                }
                b"BDC" => {
                    let tag = args.first().and_then(Obj::as_name).unwrap_or("").to_owned();
                    let p = args.get(1).cloned();
                    self.mark_begin(&tag, p.as_ref(), res);
                }
                b"EMC" => self.mark_end(),
                _ => {}
            }
            args.clear();
        }
    }

    fn xobject(&mut self, entry: Option<Obj>, res: &Dict, st: &State, depth: usize, forms: &mut Vec<u32>) {
        let Some(entry) = entry else { return };
        let num = if let Obj::Ref(n, _) = entry { Some(n) } else { None };
        let Obj::Stream(s) = self.file.resolve(&entry) else { return };
        match s.dict.name("Subtype") {
            Some("Image") => self.cover(&st.ctm),
            Some("Form") => {
                if depth >= MAX_FORMS || num.is_some_and(|n| forms.contains(&n)) {
                    return;
                }
                let data = match self.file.stream_data(&s) {
                    Ok(d) => d,
                    Err(e) => {
                        self.skip(e);
                        return;
                    }
                };
                let m = s.dict.get("Matrix").and_then(Obj::as_array).map(|a| a.iter().map(|x| x.as_num().unwrap_or(0.0)).collect::<Vec<f64>>());
                let m: Mat = match m.as_deref() {
                    Some([a, b, c, d, e, f]) => [*a, *b, *c, *d, *e, *f],
                    _ => ID,
                };
                let own = self.file.at(&s.dict, "Resources");
                let r = own.as_dict().cloned().unwrap_or_else(|| res.clone());
                let mut inner = st.clone();
                inner.ctm = mul(&m, &st.ctm);
                if let Some(n) = num {
                    forms.push(n);
                }
                self.stream(&data, &r, inner, depth + 1, forms);
                if num.is_some() {
                    forms.pop();
                }
            }
            _ => {}
        }
    }

    /// Skip an inline image: its dictionary up to `ID`, its data up to `EI`.
    fn inline_image(&mut self, lx: &mut Lexer<'_>) {
        loop {
            match lx.object() {
                Some(Err(w)) if w.as_slice() == b"ID" => break,
                None => return,
                _ => {}
            }
        }
        let d = lx.data;
        let mut p = lx.pos + 1;
        while p + 2 <= d.len() {
            if d.get(p..p + 2) == Some(b"EI")
                && d.get(p.wrapping_sub(1)).is_some_and(|b| super::lex::is_white(*b))
                && d.get(p + 2).is_none_or(|b| super::lex::is_white(*b) || super::lex::is_delim(*b))
            {
                lx.pos = p + 2;
                return;
            }
            p += 1;
        }
        lx.pos = d.len();
    }
}
