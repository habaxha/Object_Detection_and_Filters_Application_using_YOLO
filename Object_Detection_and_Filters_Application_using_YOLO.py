import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading

class YOLOObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Object Detection and Filters")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2E2E2E")

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#2E2E2E")
        self.style.configure("TButton", background="#4A4A4A", foreground="white", padding=5)
        self.style.configure("TLabel", background="#2E2E2E", foreground="white")
        self.style.configure("TNotebook", background="#2E2E2E", borderwidth=0)
        self.style.configure("TNotebook.Tab", background="#4A4A4A", foreground="white", padding=[10, 5])
        self.style.map("TNotebook.Tab", background=[("selected", "#6A6A6A")])

        self.create_widgets()

        self.weights_path = ""
        self.cfg_path = ""
        self.names_path = ""
        self.net = None
        self.classes = None
        self.output_layers = None
        self.filter_mode = None
        self.detect_objects_flag = False
        self.running = False
        self.video_capture = None
        self.minimized = {"image": False, "video": False, "live_video": False}
        self.current_image = None
        self.video_capture = None

    def create_widgets(self):
        self.main_frame = ttk.Frame(self.root, padding="10 10 10 10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.create_yolo_config_section()
        self.create_notebook()
        self.create_status_bar()

    def create_yolo_config_section(self):
        config_frame = ttk.Frame(self.main_frame, padding="5 5 5 5")
        config_frame.pack(fill=tk.X, pady=10)

        ttk.Label(config_frame, text="YOLO Configuration:").pack(side=tk.LEFT, padx=5)
        

        # Configure the style for the button
        self.style.configure("WeightsButton.TButton", font=("Segoe UI", 12), foreground="#000000", background="#FFFFFF")
        # Create the button
        weights_button = ttk.Button(config_frame, text="Select Weights", command=self.select_weights, style="WeightsButton.TButton")
        weights_button.pack(side=tk.LEFT, padx=5)
        # Add a hover effect
        self.style.map("WeightsButton.TButton", background=[("active", "#c4c4c4")])
        

        self.style.configure("CFGButton.TButton", font=("Segoe UI", 12), foreground="#000000", background="#FFFFFF")
        cfg_button = ttk.Button(config_frame, text="Select CFG", command=self.select_cfg, style="CFGButton.TButton")
        cfg_button.pack(side=tk.LEFT, padx=5)
        self.style.map("CFGButton.TButton", background=[("active", "#c4c4c4")])
        

        self.style.configure("NamesButton.TButton", font=("Segoe UI", 12), foreground="#000000", background="#FFFFFF")
        names_button = ttk.Button(config_frame, text="Select Names", command=self.select_names, style="NamesButton.TButton")
        names_button.pack(side=tk.LEFT, padx=5)
        self.style.map("NamesButton.TButton", background=[("active", "#c4c4c4")])
        

        self.style.configure("InstructionsButton.TButton", font=("Segoe UI", 12), foreground="#000000", background="#ffffff")
        instructions_button = ttk.Button(config_frame, text="Instructions", command=self.show_instructions, style="InstructionsButton.TButton")
        instructions_button.pack(side=tk.RIGHT, padx=5)
        self.style.map("InstructionsButton.TButton", background=[("active", "#c4c4c4")])
        

    def create_notebook(self):
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.image_frame = self.create_tab("Image Processing")
        self.video_frame = self.create_tab("Video Processing")
        self.live_video_frame = self.create_tab("Live Video Processing")


    def create_tab(self, title):
        frame = ttk.Frame(self.notebook, padding="10 10 10 10")
        self.notebook.add(frame, text=title)

        left_frame = ttk.Frame(frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(left_frame, bg="black", width=800, height=600)
        canvas.pack(fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        if title == "Image Processing":
            self.style.configure("ImageButton.TButton", font=("Segoe UI", 12), foreground="#FFFFFF", background="#008080")
            image_button = ttk.Button(right_frame, text="Select Image", command=self.select_image, style="ImageButton.TButton")
            image_button.pack(fill=tk.X, pady=5)
            self.style.map("ImageButton.TButton", background=[("active", "#004040")])
            self.style.configure("MinimizeButton.TButton", font=("Segoe UI", 12), foreground="#FFFFFF", background="#f1eb13")
            minimize_button = ttk.Button(right_frame, text="Minimize/Maximize", command=lambda: self.toggle_minimize("image"), style="MinimizeButton.TButton")
            minimize_button.pack(fill=tk.X, pady=5)
            self.style.map("MinimizeButton.TButton", background=[("active", "#aaa700")])
            self.style.configure("CloseButton.TButton", font=("Segoe UI", 12), foreground="#FFFFFF", background="#e73c3c")
            close_button = ttk.Button(right_frame, text="Close Image", command=self.close_image, style="CloseButton.TButton")
            close_button.pack(fill=tk.X, pady=5)
            self.style.map("CloseButton.TButton", background=[("active", "#c34242")])
        elif title == "Video Processing":
            self.style.configure("VideoButton.TButton", font=("Segoe UI", 12), foreground="#FFFFFF", background="#008080")
            video_button = ttk.Button(right_frame, text="Select Video", command=self.select_video, style="VideoButton.TButton")
            video_button.pack(fill=tk.X, pady=5)
            self.style.map("VideoButton.TButton", background=[("active", "#004040")])
            self.style.configure("MinimizeButton.TButton", font=("Segoe UI", 12), foreground="#FFFFFF", background="#f1eb13")
            minimize_button = ttk.Button(right_frame, text="Minimize/Maximize", command=lambda: self.toggle_minimize("video"), style="MinimizeButton.TButton")
            minimize_button.pack(fill=tk.X, pady=5)
            self.style.map("MinimizeButton.TButton", background=[("active", "#aaa700")])
            self.style.configure("CloseButton.TButton", font=("Segoe UI", 12), foreground="#FFFFFF", background="#e73c3c")
            close_button = ttk.Button(right_frame, text="Close Image", command=self.close_video, style="CloseButton.TButton")
            close_button.pack(fill=tk.X, pady=5)
            self.style.map("CloseButton.TButton", background=[("active", "#c34242")])
        else:
            self.style.configure("LiveVideoButton.TButton", font=("Segoe UI", 12), foreground="#FFFFFF", background="#008080")
            live_video_button = ttk.Button(right_frame, text="Start Live Video", command=self.toggle_live_video, style="LiveVideoButton.TButton")
            live_video_button.pack(fill=tk.X, pady=5)
            self.style.map("LiveVideoButton.TButton", background=[("active", "#004040")])
            self.style.configure("MinimizeButton.TButton", font=("Segoe UI", 12), foreground="#FFFFFF", background="#f1eb13")
            minimize_button = ttk.Button(right_frame, text="Minimize/Maximize", command=lambda: self.toggle_minimize("live_video"), style="MinimizeButton.TButton")
            minimize_button.pack(fill=tk.X, pady=5)
            self.style.map("MinimizeButton.TButton", background=[("active", "#aaa700")])
            self.style.configure("CloseButton.TButton", font=("Segoe UI", 12), foreground="#FFFFFF", background="#e73c3c")
            close_button = ttk.Button(right_frame, text="Close Image", command=self.close_live_video, style="CloseButton.TButton")
            close_button.pack(fill=tk.X, pady=5)
            self.style.map("CloseButton.TButton", background=[("active", "#c34242")])

        self.style.configure("DetectButton.TButton", font=("Segoe UI", 12), foreground="#FFFFFF", background="#008040")
        detect_button = ttk.Button(right_frame, text="Detect Objects", command=self.toggle_detect_objects, style="DetectButton.TButton")
        detect_button.pack(fill=tk.X, pady=5)
        self.style.map("DetectButton.TButton", background=[("active", "#004020")])
        
        filters = [
            ("Edge Detection", self.toggle_edge_detection),
            ("Sharpen", self.toggle_sharpen),
            ("Gaussian Blur", self.toggle_gaussian_blur),
            ("Brightness", self.toggle_brightness),
            ("Erosion", self.toggle_erosion),
            ("Dilation", self.toggle_dilation),
            ("Sepia Tone", self.toggle_sepia),
            ("Contrast Adjustment", self.toggle_contrast),
            ("Negative", self.toggle_negative),
            ("Emboss", self.toggle_emboss)
        ]

        self.style.configure("FilterButton.TButton", font=("Segoe UI", 12), foreground="#FFFFFF", background="#004080")
        for text, command in filters:
            filter_button = ttk.Button(right_frame, text=text, command=command, style="FilterButton.TButton")
            filter_button.pack(fill=tk.X, pady=2)
        self.style.map("FilterButton.TButton", background=[("active", "#002040")])

        return frame

    def create_status_bar(self):
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def select_weights(self):
        self.weights_path = filedialog.askopenfilename(filetypes=[("Weight files", "*.weights")])
        self.load_yolo()

    def select_cfg(self):
        self.cfg_path = filedialog.askopenfilename(filetypes=[("Config files", "*.cfg")])
        self.load_yolo()

    def select_names(self):
        self.names_path = filedialog.askopenfilename(filetypes=[("Name files", "*.names")])
        self.load_yolo()

    def load_yolo(self):
        if self.weights_path and self.cfg_path and self.names_path:
            try:
                self.net = cv2.dnn.readNet(self.weights_path, self.cfg_path)
                self.layer_names = self.net.getLayerNames()
                self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
                with open(self.names_path, "r") as f:
                    self.classes = [line.strip() for line in f.readlines()]
                messagebox.showinfo("YOLO", "YOLO model loaded successfully.")
            except Exception as e:
                messagebox.showerror("YOLO Error", f"Error loading YOLO: {e}")


    def toggle_minimize(self, section):
        self.minimized[section] = not self.minimized[section]
        if section == "image":
            self.update_image()
        elif section == "video" or section == "live_video":
            # The video loop will handle the minimized state
            pass

    def close_image(self):
        self.current_image = None
        self.clear_canvas(self.image_frame)
        self.status_var.set("Image closed")

    def close_video(self):
        self.stop_video()
        self.clear_canvas(self.video_frame)
        self.status_var.set("Video closed")

    def close_live_video(self):
        self.stop_video()
        self.clear_canvas(self.live_video_frame)
        self.status_var.set("Live video closed")

    def clear_canvas(self, frame):
        canvas = frame.winfo_children()[0].winfo_children()[0]
        canvas.delete("all")

    def stop_video(self):
        self.running = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None


    def select_image(self):
        image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if image_path:
            self.process_image(image_path)

    def process_image(self, image_path):
        self.current_image = cv2.imread(image_path)
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        self.update_image()
        self.status_var.set(f"Image loaded: {image_path}")

    def update_image(self):
        if self.current_image is not None:
            processed_image = self.apply_processing(self.current_image.copy())
            if self.minimized["image"]:
                processed_image = cv2.resize(processed_image, (200, 150))
            self.display_image(processed_image, self.image_frame)

    def select_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if video_path:
            self.stop_video()
            self.process_video(video_path)

    def process_video(self, video_path):
        self.video_capture = cv2.VideoCapture(video_path)
        self.running = True
        threading.Thread(target=self.video_loop, args=(self.video_frame, "video"), daemon=True).start()
        self.status_var.set(f"Video loaded: {video_path}")

    def toggle_live_video(self):
        if not self.running:
            self.stop_video()
            self.video_capture = cv2.VideoCapture(0)
            self.running = True
            threading.Thread(target=self.video_loop, args=(self.live_video_frame, "live_video"), daemon=True).start()
            self.status_var.set("Live video started")
        else:
            self.stop_video()
            self.status_var.set("Live video stopped")

    def video_loop(self, frame, section):
        canvas = frame.winfo_children()[0].winfo_children()[0]
        while self.running:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = self.apply_processing(frame)
            if self.minimized[section]:
                processed_frame = cv2.resize(processed_frame, (200, 150))
            self.display_image(processed_frame, canvas)
        self.clear_canvas(frame)

    def apply_processing(self, image):
        if self.detect_objects_flag and self.net is not None:
            image = self.detect_objects(image)
        if self.filter_mode:
            image = self.apply_filter(image)
        return image

    def detect_objects(self, image):
        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image

    def apply_filter(self, image):
        if self.filter_mode == "edge":
            return cv2.Canny(image, 100, 200)
        elif self.filter_mode == "sharpen":
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            return cv2.filter2D(image, -1, kernel)
        elif self.filter_mode == "gaussian_blur":
            return cv2.GaussianBlur(image, (15, 15), 0)
        elif self.filter_mode == "brightness":
            return cv2.convertScaleAbs(image, alpha=1.2, beta=50)
        elif self.filter_mode == "erosion":
            kernel = np.ones((5, 5), np.uint8)
            return cv2.erode(image, kernel, iterations=1)
        elif self.filter_mode == "dilation":
            kernel = np.ones((5, 5), np.uint8)
            return cv2.dilate(image, kernel, iterations=1)
        elif self.filter_mode == "sepia":
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            return np.clip(cv2.transform(image, sepia_filter), 0, 255).astype(np.uint8)
        elif self.filter_mode == "contrast":
            return cv2.convertScaleAbs(image, alpha=1.5, beta=0)
        elif self.filter_mode == "negative":
            return cv2.bitwise_not(image)
        elif self.filter_mode == "emboss":
            kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            return cv2.filter2D(image, -1, kernel)
        return image

    def display_image(self, image, frame):
        if isinstance(frame, ttk.Frame):
            canvas = frame.winfo_children()[0].winfo_children()[0]
        else:
            canvas = frame
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        image = Image.fromarray(image)
        image.thumbnail((canvas_width, canvas_height), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(canvas_width//2, canvas_height//2, image=photo, anchor=tk.CENTER)
        canvas.image = photo

    def toggle_detect_objects(self):
        self.detect_objects_flag = not self.detect_objects_flag
        self.update_image()

    def toggle_edge_detection(self):
        self.filter_mode = "edge" if self.filter_mode != "edge" else None
        self.update_image()

    def toggle_sharpen(self):
        self.filter_mode = "sharpen" if self.filter_mode != "sharpen" else None
        self.update_image()

    def toggle_gaussian_blur(self):
        self.filter_mode = "gaussian_blur" if self.filter_mode != "gaussian_blur" else None
        self.update_image()

    def toggle_brightness(self):
        self.filter_mode = "brightness" if self.filter_mode != "brightness" else None
        self.update_image()

    def toggle_erosion(self):
        self.filter_mode = "erosion" if self.filter_mode != "erosion" else None
        self.update_image()

    def toggle_dilation(self):
        self.filter_mode = "dilation" if self.filter_mode != "dilation" else None
        self.update_image()

    def toggle_sepia(self):
        self.filter_mode = "sepia" if self.filter_mode != "sepia" else None
        self.update_image()

    def toggle_contrast(self):
        self.filter_mode = "contrast" if self.filter_mode != "contrast" else None
        self.update_image()

    def toggle_negative(self):
        self.filter_mode = "negative" if self.filter_mode != "negative" else None
        self.update_image()

    def toggle_emboss(self):
        self.filter_mode = "emboss" if self.filter_mode != "emboss" else None
        self.update_image()

    def show_instructions(self):
        instructions_window = tk.Toplevel(self.root)
        instructions_window.title("Instructions")
        instructions_window.resizable(False, False)
        instructions_window.configure(background="#2E2E2E")  # set background to black
    
        text_widget = tk.Text(instructions_window, width=70, height=30, wrap=tk.WORD, 
                              font=("Segoe UI", 12), fg="#FFFFFF", bg="#2E2E2E")  # set font and colors
        text_widget.pack(fill=tk.BOTH, expand=True)
        instructions = """
                                            YOLO Object Detection App Instructions
        
        1. Start by loading the YOLO model:
            - Select Weights file (.weights)
            - Select Configuration file (.cfg)
            - Select Names file (.names)
            
        2. Choose a tab for the type of processing you want:
           - Image Processing
           - Video Processing
           - Live Video Processing

        3. For Image Processing:
           - Click "Select Image" to choose an image file
           - Apply filters or object detection as desired

        4. For Video Processing:
           - Click "Select Video" to choose a video file
           - Apply filters or object detection as desired

        5. For Live Video Processing:
           - Click "Start Live Video" to begin capturing from your camera
           - Apply filters or object detection as desired

        6. Use the filter buttons on the right to apply various effects

        7. Click "Detect Objects" to toggle YOLO object detection

        8. "Minimize/Maximize" and "close" the processed input as needed.

        Note: Ensure that you have loaded the YOLO model before 
        attempting object detection.
        """
        text_widget.insert(tk.INSERT, instructions)
        text_widget.config(state=tk.DISABLED)  # make the text widget read-only
    
        ok_button = tk.Button(instructions_window, text="OK", command=instructions_window.destroy, 
                      font=("Segoe UI", 12), fg="#FFFFFF", bg="#008000")  # set font and colors
        ok_button.pack(fill=tk.X, pady=10)
        
        ok_button.bind("<Enter>", lambda event: ok_button.config(bg="#004000"))  # hover effect
        ok_button.bind("<Leave>", lambda event: ok_button.config(bg="#008000"))  # return to original color
    
if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOObjectDetectionApp(root)
    root.mainloop()
    