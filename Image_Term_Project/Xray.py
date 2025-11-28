import cv2
import numpy as np

class Config:
    WINDOW_W = 945  # ความกว้างของหน้าต่างโปรแกรม (Pixel)
    WINDOW_H = 720  # ความสูงของหน้าต่างโปรแกรม (Pixel)
    SCALE_FACTOR = 1.5  # อัตราส่วนการขยายภาพตอน Crop (1.5 = ขยาย 150%)
    BLUR_KERNEL = (3, 3)  # ขนาดตัวกรองความเบลอ
    CLAHE_CLIP = 2.0  # Contrast Limit
    CLAHE_GRID = (16,16)  # Grid Size
    LOCAL_WINDOW = 11  # Window Size
    LOCAL_K0 = 2.4  # Global Mean Threshold
    LOCAL_K1 = 0.02  # Local Std Dev Threshold (Lower)
    LOCAL_K2 = 1.2 # Local Std Dev Threshold (Upper)
    LOCAL_E = 1.2  # Enhancement Factor
    TOPHAT_KERNEL = 40 # ขนาด Kernel ตัวกรองวัตถุ (ต้องใหญ่กว่าซี่โครง เล็กกว่าหัวใจ)
    FADE_STRENGTH = 0.7  # ความแรงในการลบซี่โครง (0.0 - 1.0)
    MASK_FEATHER = 21  # ยิ่งเลขเยอะ ขอบยิ่งฟุ้งกระจาย (ต้องเป็นเลขคี่)
    FEATURE_SMOOTH = 31

class ImageProcessingAlgorithms:
    # [Step 3] Local Statistical Enhancement
    @staticmethod
    def local_enhancement(img):
        window_size = Config.LOCAL_WINDOW
        k0 = Config.LOCAL_K0
        k1 = Config.LOCAL_K1
        k2 = Config.LOCAL_K2
        E = Config.LOCAL_E

        pad = window_size // 2
        padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        result = np.zeros_like(img, dtype=np.uint8)

        M_G = np.mean(img)
        D_G = np.std(img)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                local = padded[i:i + window_size, j:j + window_size]
                m_s = np.mean(local)
                sigma_s = np.std(local)

                if (m_s <= k0 * M_G) and (k1 * D_G <= sigma_s <= k2 * D_G):
                    g_val = E * img[i, j]
                    result[i, j] = np.clip(g_val, 0, 255)
                else:
                    result[i, j] = img[i, j]
        return result

    # [Step 4] Morphological Top-Hat (หาซี่โครง)
    @staticmethod
    def extract_ribs_tophat(img):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        kernel_size = Config.TOPHAT_KERNEL
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # Top-Hat Transform
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        # Normalize (0-255)
        tophat_norm = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
        return tophat_norm

    # [Step 6] Rib Suppression (Fading)
    @staticmethod
    def fade_ribs(img, tophat_map, polygon_mask):
        strength = Config.FADE_STRENGTH
        smooth_k = Config.FEATURE_SMOOTH

        img_f = img.astype(np.float32)
        tophat_f = tophat_map.astype(np.float32)
        mask_weight = polygon_mask.astype(np.float32) / 255.0

        # แก้ปัญหาเห็นเป็นวงกลมชัดๆ หรือรอยด่าง
        if smooth_k > 0:
            tophat_f = cv2.GaussianBlur(tophat_f, (smooth_k, smooth_k), 0)
        # 1. คำนวณส่วนที่จะลบ
        ribs_intensity = cv2.multiply(tophat_f, strength)
        # 2. จำกัดพื้นที่ด้วย Mask
        ribs_intensity = cv2.multiply(ribs_intensity, mask_weight)
        # 3. ลบออกจากภาพต้นฉบับ
        result = cv2.subtract(img_f, ribs_intensity)
        return np.clip(result, 0, 255).astype(np.uint8)

    # --- [NEW ALTERNATIVE] วิธีแยกความถี่ (Frequency Separation) ---
    @staticmethod
    def get_ribs_high_freq(img):
        """
        ใช้วิธี Gaussian Difference เพื่อแยก 'รายละเอียด (ซี่โครง)' ออกจาก 'โครงสร้าง (หัวใจ)'
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 1. สร้างภาพ Low Frequency (ภาพเบลอ)
        # kernel ต้องใหญ่พอที่จะทำให้ซี่โครงหายไป (เหลือแต่เงาหัวใจลางๆ)
        # ลองปรับค่า 25, 31, 41 (ต้องเป็นเลขคี่)
        blur_kernel = 41
        low_freq = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        # 2. หาภาพ High Frequency (รายละเอียด)
        # สูตร: ภาพชัด - ภาพเบลอ = รายละเอียด (ซี่โครง + ขอบหัวใจนิดหน่อย)
        # ใช้ addWeighted เพื่อจัดการเรื่องค่าติดลบให้อัตโนมัติ
        # high_freq = gray - low_freq
        high_freq = cv2.subtract(gray, low_freq)

        # 3. กรองเอาเฉพาะส่วนที่สว่าง (กระดูก)
        # เพราะ High Freq จะมีทั้งขอบมืดและขอบสว่าง เราอยากลดแค่กระดูก (ขอบสว่าง)
        # ใช้ Threshold ช่วยนิดนึงเพื่อตัด Noise พื้นหลัง
        _, ribs_map = cv2.threshold(high_freq, 15, 255, cv2.THRESH_BINARY)

        # คืนค่าเป็น Mask ที่ผ่านการกรองแล้ว (เอาไปใช้เหมือน TopHat ได้เลย)
        # แต่เพื่อความเนียน เราจะใช้ตัว High Freq ดิบๆ มาคูณกับ Mask นี้

        # ดึงเนื้อรายละเอียดซี่โครงออกมา
        ribs_detail = cv2.bitwise_and(high_freq, high_freq, mask=ribs_map)

        return ribs_detail

    # --- ฟังก์ชันลบซี่โครง (ใช้กับ High Freq ก็ได้) ---
    @staticmethod
    def fade_ribs_freq_sep(img, ribs_detail, poly_mask, strength=0.5):
        """
        ลบซี่โครงโดยใช้รายละเอียดความถี่สูง
        """
        # 1. จำกัดพื้นที่ด้วย Polygon Mask
        ribs_to_subtract = cv2.bitwise_and(ribs_detail, ribs_detail, mask=poly_mask)

        # 2. ปรับความแรง (Strength)
        ribs_to_subtract = cv2.multiply(ribs_to_subtract, strength)

        # 3. ลบออกจากภาพต้นฉบับ
        # (ภาพต้นฉบับ - รายละเอียดซี่โครง = ภาพที่ซี่โครงหายไป)
        result = cv2.subtract(img, ribs_to_subtract)

        return result
# ==========================================
# MAIN APP (ส่วนควบคุมการทำงาน)
# ==========================================
class ImageCropper:
    def __init__(self, image_path):
        self.image_path = image_path
        self.ref_point = []
        self.poly_points = []
        self.mode = "CROP"
        self.cropping = False

        # ตัวแปรเก็บภาพแต่ละ Step (เพื่อบันทึกไฟล์)
        self.step1_roi_raw = None  # ภาพดิบหลัง Crop
        self.step2_roi_smooth = None  # ภาพหลังลด Noise
        self.step3_roi_clahe = None  # ภาพหลัง CLAHE
        self.step3_roi_enhanced = None  # ภาพหลัง Local Enhance (ใช้แสดงผลตอนวาด)
        self.step4_tophat = None  # ภาพ Top-Hat
        self.step5_mask = None  # ภาพ Polygon Mask
        self.step6_result = None  # ภาพผลลัพธ์สุดท้าย

        # [Step 1] Image Acquisition
        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        self.image = self.resize_to_fit(self.image, Config.WINDOW_W, Config.WINDOW_H)
        self.clone = self.image.copy()
        self.temp_image = self.image.copy()

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.mouse_handler)
        cv2.imshow("Image", self.image)
        print("--- Step 1: Crop ---")
        print("ลากเมาส์สร้างสี่เหลี่ยมเพื่อ Crop พื้นที่ลำตัว")

    def resize_to_fit(self, img, w, h):
        h_img, w_img = img.shape[:2]
        scale = min(w / w_img, h / h_img)
        new_size = (int(w_img * scale), int(h_img * scale))
        return cv2.resize(img, new_size)

    def mouse_handler(self, event, x, y, flags, param):
        if self.mode == "CROP":
            self.handle_crop_mouse(event, x, y)
        elif self.mode == "POLYGON":
            self.handle_polygon_mouse(event, x, y)

    def handle_crop_mouse(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ref_point = [(x, y)]
            self.cropping = True
        elif event == cv2.EVENT_MOUSEMOVE and self.cropping:
            self.temp_image = self.clone.copy()
            cv2.rectangle(self.temp_image, self.ref_point[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Image", self.temp_image)
        elif event == cv2.EVENT_LBUTTONUP:
            self.ref_point.append((x, y))
            self.cropping = False
            self.perform_pipeline_part1()

    # --- Pipeline Part 1: Pre-processing & Enhancement ---
    def perform_pipeline_part1(self):
        if len(self.ref_point) == 2:
            x0, y0 = self.ref_point[0]
            x1, y1 = self.ref_point[1]
            xs, xe = sorted([x0, x1])
            ys, ye = sorted([y0, y1])

            # [Step 1] Crop ROI
            roi_raw = self.clone[ys:ye, xs:xe]
            if roi_raw.size == 0: return
            # [Step 2] Noise Reduction
            roi_smooth = cv2.GaussianBlur(roi_raw, Config.BLUR_KERNEL, 0)
            # [Step 3] Contrast Enhancement
            # 3.1 CLAHE
            clahe = cv2.createCLAHE(clipLimit=Config.CLAHE_CLIP, tileGridSize=Config.CLAHE_GRID)
            roi_clahe = clahe.apply(roi_smooth)

            # 3.2 Local Statistical Enhancement
            enhanced = ImageProcessingAlgorithms.local_enhancement(roi_clahe)

            # Resize ภาพทั้งหมดให้เท่ากัน (เพื่อความสวยงามตอนบันทึก/แสดงผล)
            h_new = int(enhanced.shape[0] * Config.SCALE_FACTOR)
            w_new = int(enhanced.shape[1] * Config.SCALE_FACTOR)

            # เก็บข้อมูลลงตัวแปร Class เพื่อใช้บันทึกทีหลัง
            self.step1_roi_raw = cv2.resize(roi_raw, (w_new, h_new))
            self.step2_roi_smooth = cv2.resize(roi_smooth, (w_new, h_new))
            self.step3_roi_clahe = cv2.resize(roi_clahe, (w_new, h_new))
            self.step3_roi_enhanced = cv2.resize(enhanced, (w_new, h_new))

            # เปลี่ยน Mode
            self.mode = "POLYGON"
            self.image = self.step3_roi_enhanced  # แสดงภาพ Enhance ให้ user วาด
            self.clone = self.image.copy()
            self.temp_image = self.image.copy()

            cv2.imshow("Image", self.image)
            print("--- Step 5: ROI Masking (Polygon) ---")
            print("1. วาดกรอบ Polygon บนภาพ Enhance")
            print("2. กด 'c' เพื่อคำนวณ")

    def handle_polygon_mouse(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.poly_points.append((x, y))
            cv2.circle(self.temp_image, (x, y), 3, (0, 0, 255), -1)
            if len(self.poly_points) > 1:
                cv2.line(self.temp_image, self.poly_points[-2], self.poly_points[-1], (0, 0, 255), 2)
            self.clone = self.temp_image.copy()
            cv2.imshow("Image", self.temp_image)

    # --- Pipeline Part 2: Feature Extraction & Suppression ---
    def run_pipeline_part2(self):
        if len(self.poly_points) < 3:
            print("Error: ต้องมีอย่างน้อย 3 จุด")
            return

        # [Step 4] Feature Extraction (Top-Hat)
        # ใช้ภาพ Enhance จาก Step 3 มาหา Top-Hat
        self.step4_tophat = ImageProcessingAlgorithms.extract_ribs_tophat(self.step3_roi_enhanced)

        # [Step 5] ROI Masking
        h, w = self.step3_roi_enhanced.shape
        self.step5_mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([self.poly_points], np.int32)
        cv2.fillPoly(self.step5_mask, pts, 255)

        k = Config.MASK_FEATHER
        self.step5_mask = cv2.GaussianBlur(self.step5_mask, (k, k), 0)

        # [Step 6] Rib Suppression (Fading)
        # เอา Top-Hat มาลบออกจากภาพ Original (Step 1)
        self.step6_result = ImageProcessingAlgorithms.fade_ribs(
            self.step1_roi_raw, self.step4_tophat, self.step5_mask
        )

        # Display & Save Results
        self.show_results(self.step4_tophat, self.step6_result, pts)
        self.save_all_steps()

    def show_results(self, feature_map, result, pts):
        show_orig = cv2.cvtColor(self.step1_roi_raw, cv2.COLOR_GRAY2BGR)
        show_map = cv2.cvtColor(feature_map, cv2.COLOR_GRAY2BGR)
        show_res = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        # วาดเส้น Polygon บน Feature Map
        cv2.polylines(show_map, [pts], True, (0, 0, 255), 2)

        combined = np.hstack((show_orig, show_map, show_res))
        if combined.shape[1] > 1200:
            scale = 1200 / combined.shape[1]
            combined = cv2.resize(combined, (0, 0), fx=scale, fy=scale)

        cv2.imshow("Pipeline Result: Original | TopHat Map | Final Faded", combined)

    def save_all_steps(self):
        cv2.imwrite("monkey1/step1_cropped_raw.jpg", self.step1_roi_raw)
        cv2.imwrite("monkey1/step2_noise_reduction.jpg", self.step2_roi_smooth)
        cv2.imwrite("monkey1/step3_1_contrast_clahe.jpg", self.step3_roi_clahe)
        cv2.imwrite("monkey1/step3_2_enhanced.jpg", self.step3_roi_enhanced)
        cv2.imwrite("monkey1/step4_tophat_feature.jpg", self.step4_tophat)
        cv2.imwrite("monkey1/step5_roi_mask.jpg", self.step5_mask)
        cv2.imwrite("monkey1/step6_final_result.jpg", self.step6_result)

    def run(self):
        while True:
            key = cv2.waitKey(20) & 0xFF
            if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1: break
            if key == 27 or key == ord('q'):
                break
            elif key == ord('r'):  # Reset
                self.image = self.resize_to_fit(cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE), Config.WINDOW_W,
                                                Config.WINDOW_H)
                self.clone = self.image.copy()
                self.temp_image = self.image.copy()
                self.ref_point = []
                self.poly_points = []
                self.mode = "CROP"
                self.cropping = False

                # Reset variables
                self.step1_roi_raw = None
                self.step2_roi_smooth = None
                self.step3_roi_clahe = None
                self.step3_roi_enhanced = None
                self.step4_tophat = None
                self.step5_mask = None
                self.step6_result = None

                cv2.destroyAllWindows()
                cv2.namedWindow("Image")
                cv2.setMouseCallback("Image", self.mouse_handler)
                cv2.imshow("Image", self.image)
                print("--- Step 1: Crop ---")
            elif key == ord('c'):
                if self.mode == "POLYGON":
                    self.run_pipeline_part2()
                else:
                    print("กรุณา Crop ก่อน")
        cv2.destroyAllWindows()
if __name__ == "__main__":
    try:
        cropper = ImageCropper("monkey1.jpg")
        cropper.run()
    except FileNotFoundError as e:
        print(e)