"""Generate promotional images for Stockscope in 3 sizes."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# --- Brand palette ---
NAVY_DEEP = (5, 8, 24)
NAVY = (8, 11, 34)
NAVY_LIGHT = (26, 31, 61)
CHARTREUSE = (190, 242, 100)
CHARTREUSE_BRIGHT = (217, 249, 157)
BLUE_ELEC = (59, 130, 246)
BLUE_DEEP = (30, 64, 175)
WHITE = (240, 247, 255)
MUTED = (148, 168, 212)

# --- Font paths ---
FONT_REG = "C:\\Windows\\Fonts\\LeelawUI.ttf"
FONT_BOLD = "C:\\Windows\\Fonts\\LeelaUIb.ttf"
FONT_MONO = "C:\\Windows\\Fonts\\consola.ttf"


def aurora_bg(w: int, h: int) -> Image.Image:
    """Generate aurora gradient background using radial blends."""
    img = np.zeros((h, w, 3), dtype=np.float32)
    # base navy
    img[:, :, 0] = NAVY_DEEP[0]
    img[:, :, 1] = NAVY_DEEP[1]
    img[:, :, 2] = NAVY_DEEP[2]

    # blob list: (cx, cy, radius_x, radius_y, color, intensity)
    blobs = [
        # Blue dominant
        (w * 0.8, h * 0.0, w * 0.7, h * 0.5, BLUE_ELEC, 0.55),
        (w * 0.15, h * 0.5, w * 0.6, h * 0.45, BLUE_ELEC, 0.45),
        (w * 0.5, h * 1.0, w * 0.55, h * 0.4, BLUE_DEEP, 0.45),
        # Chartreuse highlights
        (w * 0.25, h * 0.30, w * 0.40, h * 0.18, CHARTREUSE_BRIGHT, 0.40),
        (w * 0.75, h * 0.55, w * 0.30, h * 0.15, CHARTREUSE, 0.35),
        (w * 0.10, h * 0.85, w * 0.45, h * 0.20, CHARTREUSE_BRIGHT, 0.30),
        (w * 0.88, h * 0.20, w * 0.22, h * 0.12, CHARTREUSE, 0.28),
    ]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for cx, cy, rx, ry, color, intensity in blobs:
        # elliptical falloff
        d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        falloff = np.clip(1.0 - d, 0, 1) ** 2 * intensity
        for i in range(3):
            img[:, :, i] = img[:, :, i] + (color[i] - img[:, :, i]) * falloff
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def draw_logo(draw: ImageDraw.ImageDraw, x: int, y: int, size: int):
    """Draw the rounded square 'S' logo."""
    # Logo bg square with gradient (approximate with solid)
    pad = size // 8
    draw.rounded_rectangle(
        [(x, y), (x + size, y + size)],
        radius=size // 5,
        fill=NAVY_LIGHT,
        outline=CHARTREUSE,
        width=max(2, size // 28),
    )
    f = ImageFont.truetype(FONT_BOLD, int(size * 0.7))
    bbox = draw.textbbox((0, 0), "S", font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x + (size - tw) / 2 - bbox[0], y + (size - th) / 2 - bbox[1]), "S", font=f, fill=CHARTREUSE)


def make_square(out: str = "promo_square.png"):
    W, H = 1080, 1080
    img = aurora_bg(W, H)
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    draw = ImageDraw.Draw(img)

    # Logo + brand at top
    logo_size = 140
    draw_logo(draw, 80, 100, logo_size)
    f_brand = ImageFont.truetype(FONT_BOLD, 76)
    f_sub = ImageFont.truetype(FONT_BOLD, 28)
    draw.text((240, 120), "Stockscope", font=f_brand, fill=WHITE)
    draw.text((242, 200), "REAL-TIME · GLOBAL", font=f_sub, fill=CHARTREUSE)

    # Headline
    f_h1 = ImageFont.truetype(FONT_BOLD, 88)
    f_h2 = ImageFont.truetype(FONT_BOLD, 64)
    draw.text((80, 330), "แดชบอร์ดหุ้น", font=f_h1, fill=WHITE)
    draw.text((80, 440), "วิเคราะห์เรียลไทม์", font=f_h1, fill=CHARTREUSE)

    # Features
    f_feat = ImageFont.truetype(FONT_REG, 36)
    features = [
        "◆  กราฟ TradingView + แนวรับ/แนวต้านอัตโนมัติ",
        "◆  ข่าวอัพเดต + แปลไทย + AI สรุป",
        "◆  สแกนหุ้น · Pre-Market · Watchlist",
        "◆  หุ้น US · คริปโต · ไทย · ทั่วโลก",
        "◆  สัญญาณซื้อ-ขาย + Risk/Reward",
    ]
    y = 600
    for ft in features:
        # diamond bullet in chartreuse
        f_b = ImageFont.truetype(FONT_BOLD, 36)
        draw.text((80, y), "●", font=f_b, fill=CHARTREUSE)
        draw.text((130, y), ft[3:].strip(), font=f_feat, fill=WHITE)
        y += 60

    # URL + CTA
    f_url = ImageFont.truetype(FONT_BOLD, 38)
    f_cta = ImageFont.truetype(FONT_BOLD, 42)
    url_text = "stockscopeproinw.streamlit.app"
    bbox = draw.textbbox((0, 0), url_text, font=f_url)
    uw = bbox[2] - bbox[0]
    draw.rounded_rectangle(
        [(80, 940), (80 + uw + 60, 1010)],
        radius=35,
        fill=CHARTREUSE,
    )
    draw.text((110, 952), url_text, font=f_url, fill=NAVY_DEEP)
    draw.text((80 + uw + 100, 950), "ทดลองฟรี ›", font=f_cta, fill=CHARTREUSE)

    img.save(out, "PNG", optimize=True)
    print(f"Saved {out}")


def make_story(out: str = "promo_story.png"):
    W, H = 1080, 1920
    img = aurora_bg(W, H)
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    draw = ImageDraw.Draw(img)

    # Top area: logo + brand centered
    logo_size = 180
    draw_logo(draw, (W - logo_size) // 2, 220, logo_size)
    f_brand = ImageFont.truetype(FONT_BOLD, 110)
    bbox = draw.textbbox((0, 0), "Stockscope", font=f_brand)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, 440), "Stockscope", font=f_brand, fill=WHITE)
    f_sub = ImageFont.truetype(FONT_BOLD, 36)
    sub_text = "REAL-TIME · GLOBAL"
    bbox = draw.textbbox((0, 0), sub_text, font=f_sub)
    sw = bbox[2] - bbox[0]
    draw.text(((W - sw) // 2, 560), sub_text, font=f_sub, fill=CHARTREUSE)

    # Headline centered
    f_h1 = ImageFont.truetype(FONT_BOLD, 100)
    h1_text = "วิเคราะห์หุ้น"
    bbox = draw.textbbox((0, 0), h1_text, font=f_h1)
    hw = bbox[2] - bbox[0]
    draw.text(((W - hw) // 2, 720), h1_text, font=f_h1, fill=WHITE)
    h2_text = "แบบเรียลไทม์"
    bbox = draw.textbbox((0, 0), h2_text, font=f_h1)
    hw = bbox[2] - bbox[0]
    draw.text(((W - hw) // 2, 840), h2_text, font=f_h1, fill=CHARTREUSE)

    # Feature cards (centered)
    f_feat = ImageFont.truetype(FONT_BOLD, 42)
    features = [
        "กราฟ TradingView ของจริง",
        "แนวรับ/แนวต้านอัตโนมัติ",
        "ข่าวแปลไทย + AI สรุป",
        "Pre-Market · Watchlist · Screener",
        "หุ้น US · คริปโต · ไทย",
    ]
    y = 1080
    for ft in features:
        bbox = draw.textbbox((0, 0), ft, font=f_feat)
        fw = bbox[2] - bbox[0]
        # card bg
        card_w = fw + 80
        card_x = (W - card_w) // 2
        draw.rounded_rectangle(
            [(card_x, y), (card_x + card_w, y + 75)],
            radius=15,
            fill=(*NAVY_LIGHT, 200) if len(NAVY_LIGHT) == 4 else NAVY_LIGHT,
            outline=CHARTREUSE,
            width=2,
        )
        draw.text((card_x + 40, y + 12), ft, font=f_feat, fill=WHITE)
        y += 100

    # CTA bottom
    f_cta = ImageFont.truetype(FONT_BOLD, 60)
    cta_text = "ทดลองใช้ฟรี ›"
    bbox = draw.textbbox((0, 0), cta_text, font=f_cta)
    cw = bbox[2] - bbox[0]
    pill_x = (W - cw - 100) // 2
    draw.rounded_rectangle(
        [(pill_x, 1680), (pill_x + cw + 100, 1780)],
        radius=50,
        fill=CHARTREUSE,
    )
    draw.text((pill_x + 50, 1695), cta_text, font=f_cta, fill=NAVY_DEEP)

    # URL at very bottom
    f_url = ImageFont.truetype(FONT_BOLD, 36)
    url_text = "stockscopeproinw.streamlit.app"
    bbox = draw.textbbox((0, 0), url_text, font=f_url)
    uw = bbox[2] - bbox[0]
    draw.text(((W - uw) // 2, 1820), url_text, font=f_url, fill=WHITE)

    img.save(out, "PNG", optimize=True)
    print(f"Saved {out}")


def make_banner(out: str = "promo_banner.png"):
    W, H = 1600, 900
    img = aurora_bg(W, H)
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    draw = ImageDraw.Draw(img)

    # Logo + brand top-left
    logo_size = 130
    draw_logo(draw, 80, 80, logo_size)
    f_brand = ImageFont.truetype(FONT_BOLD, 72)
    f_sub = ImageFont.truetype(FONT_BOLD, 26)
    draw.text((230, 92), "Stockscope", font=f_brand, fill=WHITE)
    draw.text((232, 175), "REAL-TIME · GLOBAL", font=f_sub, fill=CHARTREUSE)

    # Big headline center-left
    f_h1 = ImageFont.truetype(FONT_BOLD, 110)
    draw.text((80, 290), "วิเคราะห์หุ้น", font=f_h1, fill=WHITE)
    draw.text((80, 410), "เรียลไทม์ ทั่วโลก", font=f_h1, fill=CHARTREUSE)

    # Features right side compact
    f_feat = ImageFont.truetype(FONT_BOLD, 32)
    features = [
        "◆ กราฟ TradingView + แนวรับ/ต้าน Auto",
        "◆ ข่าวแปลไทย + AI สรุป",
        "◆ Screener · Pre-Market · Watchlist",
        "◆ หุ้น US · คริปโต · ไทย",
    ]
    y = 310
    f_b = ImageFont.truetype(FONT_BOLD, 32)
    for ft in features:
        draw.text((900, y), "●", font=f_b, fill=CHARTREUSE)
        draw.text((930, y), ft[2:].strip(), font=f_feat, fill=WHITE)
        y += 55

    # URL + CTA bottom
    f_url = ImageFont.truetype(FONT_BOLD, 36)
    f_cta = ImageFont.truetype(FONT_BOLD, 44)
    url_text = "stockscopeproinw.streamlit.app"
    bbox = draw.textbbox((0, 0), url_text, font=f_url)
    uw = bbox[2] - bbox[0]
    draw.rounded_rectangle(
        [(80, 770), (80 + uw + 60, 840)],
        radius=35,
        fill=CHARTREUSE,
    )
    draw.text((110, 782), url_text, font=f_url, fill=NAVY_DEEP)
    draw.text((W - 350, 780), "ทดลองฟรี ›", font=f_cta, fill=CHARTREUSE)

    img.save(out, "PNG", optimize=True)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_square("promo_square.png")
    make_story("promo_story.png")
    make_banner("promo_banner.png")
    print("\nDone! 3 images created:")
    print("  - promo_square.png  (1080x1080)  IG post")
    print("  - promo_story.png   (1080x1920)  IG/TikTok story")
    print("  - promo_banner.png  (1600x900)   Twitter/FB cover")
