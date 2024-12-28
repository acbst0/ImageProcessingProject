import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageEnhance, ImageOps, ImageFilter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RangeSlider

def create_range_slider(frame):
    """
    Matplotlib kullanarak küçük ve kompakt bir çift çubuklu slider oluşturur.
    """
    # Daha küçük boyutlar
    fig = Figure(figsize=(5, 1), dpi=50)  # Daha kompakt bir slider
    ax = fig.add_subplot(111) # Daha küçük başlık yazı boyutu
    ax.set_facecolor("lightgray")  # Gri arka plan

    global range_slider
    range_slider = RangeSlider(ax, "", 0, 255, valinit=(50, 150))
    range_slider.label.set_fontsize(2)  # Etiket yazı boyutunu küçült

    # Tkinter içine yerleştir
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.get_tk_widget().pack()

def create_form(image_path):
    # Ana pencereyi oluştur
    root = tk.Tk()
    root.title("Image Processing")
    root.geometry("1300x600")  # Pencere boyutu büyütüldü
    root.configure(bg="white")

    # Ana çerçeveyi oluştur (Grid sistemi)
    style = ttk.Style()
    style.configure("White.TFrame", background="white")  # Beyaz arka plan stili

    frame = ttk.Frame(root, style="White.TFrame")  # Stili uygula
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Resim çerçevesi (Sol tarafta)
    image_frame = ttk.Frame(frame)
    image_frame.grid(row=0, column=0, sticky="n", padx=10)

    # Reset Butonu ikisinin de altında
    frame_Reset_but = ttk.Frame(frame, style="White.TFrame")
    frame_Reset_but.grid(row=1,column=0, columnspan=3, sticky="w", padx=10)

    # Kontroller çerçevesi (Sağ tarafta)
    control_frame = ttk.Frame(frame)
    control_frame.grid(row=0, column=1, sticky="n", padx=10)

    frameBright = ttk.Frame(control_frame)
    frameBright.grid(row=0, column=0, sticky='n', padx=10)
    
    frameThres = ttk.Frame(control_frame)
    frameThres.grid(row=0, column=1, sticky='n', padx=10)

    frameThresBut = ttk.Frame(control_frame)
    frameThresBut.grid(row=0, column=2, sticky='w', padx=10)

    frameNeg = ttk.Frame(control_frame)
    frameNeg.grid(row=1, column=0, sticky='n', padx=10)

    frameHistogram = ttk.Frame(control_frame)
    frameHistogram.grid(row=1, column=1, sticky='n', padx=10)

    frameEdge = ttk.Frame(control_frame)
    frameEdge.grid(row=1, column=2, sticky='w', padx=10)

    frameFiltersLineer = ttk.Frame(control_frame)
    frameFiltersLineer.grid(row=2, column=0, sticky='n', padx=10)

    frameFiltersMedian = ttk.Frame(control_frame)
    frameFiltersMedian.grid(row=2, column=1, sticky='n', padx=10)

    frameHistMatch = ttk.Frame(control_frame)
    frameHistMatch.grid(row=3, column=0, sticky='n', padx=10)

    framePower = ttk.Frame(control_frame)
    framePower.grid(row=3, column=1, sticky='n', padx=10)

    framePowerBut = ttk.Frame(control_frame)
    framePowerBut.grid(row=3, column=2, sticky='w', padx=10)

    frameLogTrans = ttk.Frame(control_frame)
    frameLogTrans.grid(row=4, column=0, sticky='n', padx=10)

    frameBitSlice = ttk.Frame(control_frame)
    frameBitSlice.grid(row=4, column=1, sticky='n', padx=10)

    frameBitSliceBut = ttk.Frame(control_frame)
    frameBitSliceBut.grid(row=4, column=2, sticky='w', padx=10)

    frameMorphologic = ttk.Frame(control_frame)
    frameMorphologic.grid(row=5, column=0, sticky="n", padx=0)

    frameCanny = ttk.Frame(control_frame, style="White.TFrame")
    frameCanny.grid(row=5, column=1, sticky="n", padx=10)

    frameCannyBut = ttk.Frame(control_frame)
    frameCannyBut.grid(row=5, column=2, sticky="w", padx=10)

    frameHough = ttk.Frame(control_frame)
    frameHough.grid(row=7,column=0, sticky="n", padx=10)

    frameConnectedComp = ttk.Frame(control_frame)
    frameConnectedComp.grid(row=7,column=1, sticky="n", padx=10)

    frameGrey = ttk.Frame(control_frame)
    frameGrey.grid(row=8,column=0, sticky="n", padx=10)

    frameGreyTwo = ttk.Frame(control_frame)
    frameGreyTwo.grid(row=8,column=1, sticky="n", padx=10)

    frameGreyThree = ttk.Frame(control_frame)
    frameGreyThree.grid(row=8,column=2, sticky="w", padx=10)

    # Resmi yükle ve göster
    try:
        original_image = Image.open(image_path)
        original_image = original_image.resize((400, 400), Image.ANTIALIAS)  # Resim boyutu büyütüldü

        #***************************************************************************

        # Reset Fonksiyonu
        def resetBut():
            photo = ImageTk.PhotoImage(original_image)
            label.config(image=photo)
            label.image = photo

        # Parlaklığı ayarlayacak bir fonksiyon
        def update_brightness(value):
            enhancer = ImageEnhance.Brightness(original_image)
            bright_image = enhancer.enhance(float(value))
            photo = ImageTk.PhotoImage(bright_image)
            label.config(image=photo)
            label.image = photo  # Referansı sakla

        # Eşikleme işlemi için fonksiyon
        def apply_threshold():
            threshold_value = threshold_slider.get()
            grayscale_image = original_image.convert("L")  # Gri tonlama
            threshold_image = grayscale_image.point(lambda p: 255 if p > threshold_value else 0)
            photo = ImageTk.PhotoImage(threshold_image)
            label.config(image=photo)
            label.image = photo  # Referansı sakla

        # Negatif görüntü için fonksiyon
        def toggle_negative():
            if negative_var.get():
                negative_image = Image.eval(original_image, lambda p: 255 - p)
                photo = ImageTk.PhotoImage(negative_image)
            else:
                photo = ImageTk.PhotoImage(original_image)
            label.config(image=photo)
            label.image = photo  # Referansı sakla

        # Histogram eşitleme için fonksiyon
        def toggle_histogram_equal():
            if histogram_var.get():
                grayscale_image = original_image.convert("L")  # Gri tonlama
                equalized_image = ImageOps.equalize(grayscale_image)  # Histogram eşitleme
                photo = ImageTk.PhotoImage(equalized_image)
            else:
                photo = ImageTk.PhotoImage(original_image)
            label.config(image=photo)
            label.image = photo  # Referansı sakla

        # Kenar algılama fonksiyonu
        def apply_edge_detection():
            edge_image = original_image.filter(ImageFilter.FIND_EDGES)  # Kenar algılama
            photo = ImageTk.PhotoImage(edge_image)
            label.config(image=photo)
            label.image = photo  # Referansı sakla

        # Lineer ve nonlineer filtreleme fonksiyonları
        def apply_filter(selected_filter):
            if selected_filter == "Gaussian Blur":
                filtered_image = original_image.filter(ImageFilter.GaussianBlur(2))
            elif selected_filter == "Box Blur":
                filtered_image = original_image.filter(ImageFilter.BoxBlur(2))
            elif selected_filter == "Median Filter":
                filtered_image = original_image.filter(ImageFilter.MedianFilter(size=3))
            elif selected_filter == "Min Filter":
                filtered_image = original_image.filter(ImageFilter.MinFilter(size=3))
            elif selected_filter == "Max Filter":
                filtered_image = original_image.filter(ImageFilter.MaxFilter(size=3))
            elif selected_filter == "None":
                filtered_image = original_image
            else:
                filtered_image = original_image
            
            photo = ImageTk.PhotoImage(filtered_image)
            label.config(image=photo)
            label.image = photo  # Referansı sakla

        # Histogram Matching
        def histogramMatching():
            if histoMatch.get():
                original_image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                grayscale_image = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2GRAY)
                equalized_image = cv2.equalizeHist(grayscale_image)
                equalized_image_rgb = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB)
                photo = ImageTk.PhotoImage(image=Image.fromarray(equalized_image_rgb))
            else:
                photo = ImageTk.PhotoImage(original_image)

            label.config(image=photo)
            label.image = photo  # Referansı sakla

        def apply_power_transform():
            gamma = gamma_slider.get()  # Slider'dan seçilen gamma değeri

            # Orijinal görüntüyü OpenCV formatına dönüştür
            original_image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

            # Gri tonlama (isteğe bağlı)
            grayscale_image = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2GRAY)

            # Güç dönüşümü
            c = 255 / (np.max(grayscale_image) ** gamma)  # Normalizasyon sabiti
            power_transformed = c * (grayscale_image.astype(np.float32) ** gamma)

            # Görüntüyü 8-bit formatına dönüştür
            power_transformed = np.uint8(power_transformed)

            # Görüntüyü Tkinter uyumlu hale getir
            power_transformed_rgb = cv2.cvtColor(power_transformed, cv2.COLOR_GRAY2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(power_transformed_rgb))

            # Arayüzde güncelleme
            label.config(image=photo)
            label.image = photo  # Referansı sakla

        def log_trans_f():
            if logTrans.get():
                original_image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                grayscale_image = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2GRAY)
                c = 255 / np.log(1 + np.max(grayscale_image))  # Normalizasyon için sabit
                log_transformed = c * np.log(1 + grayscale_image.astype(np.float32))
                log_transformed = np.uint8(log_transformed)
                log_transformed_rgb = cv2.cvtColor(log_transformed, cv2.COLOR_GRAY2RGB)
                photo = ImageTk.PhotoImage(image=Image.fromarray(log_transformed_rgb))
            else:
                photo = ImageTk.PhotoImage(original_image)

            label.config(image=photo)
            label.image = photo

        def applyBitSlicing():
            bit_plane = bit_slider.get()
            original_image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
            grayscale_image = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2GRAY)
            bit_mask = 1 << bit_plane
            sliced_image = cv2.bitwise_and(grayscale_image, bit_mask)
            sliced_image = (sliced_image > 0) * 255
            sliced_image_rgb = cv2.cvtColor(sliced_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(sliced_image_rgb))
            label.config(image=photo)
            label.image = photo

        def apply_morphological_operation(operation_type):
            """
            Morfolojik işlem uygular.

            Parametre:
                operation_type: Seçilen işlem türü (erode, dilate, opening, closing).
            """
            # Orijinal görüntüyü OpenCV formatına dönüştür
            original_image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

            # Gri tonlama
            grayscale_image = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2GRAY)

            # Yapılandırma elemanı (kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 5x5 dikdörtgen kernel

            # İşlem türüne göre morfolojik işlemi seç
            if operation_type == "Erode":
                processed_image = cv2.erode(grayscale_image, kernel, iterations=1)
            elif operation_type == "Dilate":
                processed_image = cv2.dilate(grayscale_image, kernel, iterations=1)
            elif operation_type == "Opening":
                processed_image = cv2.morphologyEx(grayscale_image, cv2.MORPH_OPEN, kernel)
            elif operation_type == "Closing":
                processed_image = cv2.morphologyEx(grayscale_image, cv2.MORPH_CLOSE, kernel)
            elif operation_type == "None":
                processed_image = original_image
                photo = ImageTk.PhotoImage(processed_image)
                label.config(image=photo)
                label.image = photo  # Referansı sakla
                return 
            else:
                processed_image = original_image
                photo = ImageTk.PhotoImage(processed_image)
                label.config(image=photo)
                label.image = photo  # Referansı sakla
                return
            
            # Görüntüyü Tkinter uyumlu hale getir
            processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(processed_image_rgb))

            # Arayüzde güncelleme
            label.config(image=photo)
            label.image = photo  # Referansı sakla

        def apply_hough_transform_opencv():
            """
            OpenCV kullanarak Hough dönüşümünü uygular.
            Checkbox durumu dikkate alınır.
            """
            if houghvar.get():  # Checkbox işaretli mi?
                # Görüntüyü numpy formatına dönüştür
                image_array = np.array(original_image)

                # Gri tonlama
                grayscale_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

                # Kenar algılama (Canny)
                edges = cv2.Canny(grayscale_image, 50, 150, apertureSize=3)

                # Doğrular için Hough dönüşümü
                lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0,0)

                # Tespit edilen doğruları işaretleme
                hough_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
                if lines is not None:
                    for rho, theta in lines[:, 0]:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * a)
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * a)
                        cv2.line(hough_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Kırmızı doğrular

                # Tkinter için dönüştür
                hough_image_pil = Image.fromarray(cv2.cvtColor(hough_image, cv2.COLOR_BGR2RGB))
                photo = ImageTk.PhotoImage(hough_image_pil)
            else:
                # Orijinal görüntüyü Tkinter uyumlu hale getir
                photo = ImageTk.PhotoImage(image=original_image)

            # Arayüzde güncelleme
            label.config(image=photo)
            label.image = photo  # Referansı sakla

        ## Canny Edge Detection
        def apply_canny_edge_detection():
            """
            Canny kenar algılama için alt ve üst eşik değerlerini tek bir slider'dan alır.
            """
            lower_threshold, upper_threshold = range_slider.val
            # Görüntüyü numpy formatına dönüştür
            image_array = np.array(original_image)
            # Gri tonlama
            grayscale_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            # Kenar algılama (Canny)
            edges = cv2.Canny(grayscale_image, int(lower_threshold), int(upper_threshold))
            # Tkinter için dönüştür
            edges_pil = Image.fromarray(edges)
            photo = ImageTk.PhotoImage(edges_pil)

            # Arayüzde güncelleme
            label.config(image=photo)
            label.image = photo
            
            
            ## The things that i will add
            ## Connected Component Analysis
        def apply_connected_component_analysis():
            """
            Bağlı Bileşen Analizi (Connected Component Analysis) uygular.
            Checkbox durumu dikkate alınır.
            """
            if connectedvar.get():  # Checkbox işaretli mi?
                # Görüntüyü numpy dizisine dönüştür
                image_array = np.array(original_image)

                # Gri tonlama
                grayscale_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

                # İkili görüntüye dönüştür (Otsu eşikleme kullanılarak)
                _, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Bağlı bileşen analizi
                num_labels, labels_image = cv2.connectedComponents(binary_image)

                # Renkli bir sonuç oluştur
                label_hue = np.uint8(179 * labels_image / np.max(labels_image))  # Her bileşen için farklı renk
                blank_channel = 255 * np.ones_like(label_hue)  # Alfa kanalı
                labeled_image = cv2.merge([label_hue, blank_channel, blank_channel])

                # HSV'den RGB'ye dönüştür
                labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_HSV2RGB)

                # Etiket değeri sıfır olan (arka plan) pikselleri beyaz yap
                labeled_image[label_hue == 0] = 255

                # Tkinter uyumlu hale getir
                photo = ImageTk.PhotoImage(image=Image.fromarray(labeled_image))
            else:
                # Orijinal görüntüyü Tkinter uyumlu hale getir
                photo = ImageTk.PhotoImage(image=original_image)

            # Arayüzde görüntüyü güncelle
            label.config(image=photo)
            label.image = photo  # Referansı sakla

        ## Grey Level Transformations
        def apply_grey_level_transformations():
            """
            Gri Tonlama Dönüşümleri (Grey Level Transformations) uygular.
            Slider'dan alınan parlaklık ve kontrast değerlerine göre işlemi gerçekleştirir.
            """
            # Parlaklık ve kontrast değerlerini slider'dan al
            brightness = br_scale.get()
            contrast = contrast_slider.get()

            # Görüntüyü numpy dizisine dönüştür
            image_array = np.array(original_image)

            # Gri tonlama
            grayscale_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

            # Parlaklık ve kontrast uygulama
            transformed_image = cv2.convertScaleAbs(grayscale_image, alpha=contrast / 100, beta=brightness)

            # Tkinter için uygun formata dönüştür
            photo = ImageTk.PhotoImage(image=Image.fromarray(transformed_image))

            # Arayüzde görüntüyü güncelle
            label.config(image=photo)
            label.image = photo  # Referansı sakla



        #***************************************************************************

        # Orijinal resmi göster
        photo = ImageTk.PhotoImage(original_image)
        label = tk.Label(image_frame, image=photo)
        label.image = photo  # Referansı sakla
        label.pack()

        # Reset Butonu
        ResetBUTTON = ttk.Button(frame_Reset_but, text="Reset", command=resetBut)
        ResetBUTTON.pack(pady=10)
        # Parlaklık için slider ekle
        brightness_slider = tk.Scale(frameBright, from_=0.5, to=2.0, 
                                      resolution=0.1, orient=tk.HORIZONTAL, 
                                      label="Brightness", command=update_brightness)
        brightness_slider.set(1.0)  # Varsayılan parlaklık değeri
        brightness_slider.pack(pady=10)

        # Eşikleme için slider ve buton ekle
        threshold_slider = tk.Scale(frameThres, from_=0, to=255, 
                                     resolution=1, orient=tk.HORIZONTAL, 
                                     label="Threshold")
        threshold_slider.set(128)  # Varsayılan eşik değeri
        threshold_slider.pack(pady=10)

        threshold_button = ttk.Button(frameThresBut, text="Apply Threshold", command=apply_threshold)
        threshold_button.pack(pady=10)

        # Negatif görüntü için checkbox ekle
        negative_var = tk.BooleanVar()
        negative_checkbox = ttk.Checkbutton(frameNeg, text="Negative Image", variable=negative_var, command=toggle_negative)
        negative_checkbox.pack(pady=10)

        # Histogram eşitleme için checkbox ekle
        histogram_var = tk.BooleanVar()
        histogram_checkbox = ttk.Checkbutton(frameHistogram, text="Histogram Equalization", variable=histogram_var, command=toggle_histogram_equal)
        histogram_checkbox.pack(pady=10)

        # Kenar algılama için buton ekle
        edge_button = ttk.Button(frameEdge, text="Edge Detection", command=apply_edge_detection)
        edge_button.pack(pady=10)

        # Lineer filtreleme için dropdown menü ekle
        linear_label = tk.Label(frameFiltersLineer, text="Linear Filters")
        linear_filters = ["None", "None", "Gaussian Blur", "Box Blur"]
        linear_var = tk.StringVar(value="None")
        linear_menu = ttk.OptionMenu(frameFiltersLineer, linear_var, *linear_filters, command=apply_filter)
        linear_menu.pack(pady=5)
        linear_label.pack()

        # Nonlineer filtreleme için dropdown menü ekle
        nonlinear_label = tk.Label(frameFiltersMedian, text="Nonlinear Filters")
        nonlinear_filters = ["None", "None", "Median Filter", "Min Filter", "Max Filter"]
        nonlinear_var = tk.StringVar(value="None")
        nonlinear_menu = ttk.OptionMenu(frameFiltersMedian, nonlinear_var, *nonlinear_filters, command=apply_filter)
        nonlinear_menu.pack(pady=5)
        nonlinear_label.pack()

        # Histogram Matching checkbox
        histoMatch = tk.BooleanVar()
        histogramMatch_checkbox = ttk.Checkbutton(frameHistMatch, text="Histogram Matching", variable=histoMatch, command=histogramMatching)
        histogramMatch_checkbox.pack(pady=10)

        # Power Law
        gamma_slider = tk.Scale(framePower, from_=0.1, to=5.0,  # Gamma aralığı
                        resolution=0.1, orient=tk.HORIZONTAL, 
                        label="Gamma Value")
        gamma_slider.set(1.0)  # Varsayılan gamma değeri
        gamma_slider.pack(pady=10)
        gamma_button = ttk.Button(framePowerBut, text="Apply Power Transform", command=apply_power_transform)
        gamma_button.pack(pady=10)

        # Log Transformation
        logTrans = tk.BooleanVar()
        logTrans_checkbox = ttk.Checkbutton(frameLogTrans, text="Logarithmic Transformation", variable=logTrans, command=log_trans_f)
        logTrans_checkbox.pack(pady=10)

        # Bit Plane Slicing
        bit_slider = tk.Scale(frameBitSlice, from_=0, to=7,  # Gamma aralığı
                        resolution=1, orient=tk.HORIZONTAL, 
                        label="Bit Value")
        bit_slider.set(1.0)  # Varsayılan gamma değeri
        bit_slider.pack(pady=10)
        bit_button = ttk.Button(frameBitSliceBut, text="Bit Plane Slicing", command=applyBitSlicing)
        bit_button.pack(pady=10)

        # Morphological Filtering
        morph_label = tk.Label(frameMorphologic, text="Morphological Filters")
        morph_filters = ["None", "None", "Erode", "Dilate", "Opening", "Closing"]
        morph_var = tk.StringVar(value="None")
        morph_menu = ttk.OptionMenu(frameMorphologic, morph_var, *morph_filters, command=apply_morphological_operation)
        morph_menu.pack(pady=5)
        morph_label.pack()

        # Hough Transform
        houghvar = tk.BooleanVar()
        hough_checkbox = ttk.Checkbutton(frameHough, text="Hough Transform", variable=houghvar, command=apply_hough_transform_opencv)
        hough_checkbox.pack(pady=10)

        # Canny Edge Detection
        create_range_slider(frameCanny)

        apply_button = ttk.Button(frameCannyBut, text="Apply Canny Edge Detection", command=apply_canny_edge_detection)
        apply_button.pack(pady=10)

        # Connected Component Analysis
        connectedvar = tk.BooleanVar()
        connectedvars_checkbox = ttk.Checkbutton(frameConnectedComp, text="Connected Component Analysis", variable=connectedvar, command=apply_connected_component_analysis)
        connectedvars_checkbox.pack(pady=10)

        # Grey Scale
        br_scale = tk.Scale(frameGrey, from_=-100, to=100, orient=tk.HORIZONTAL, label="Brightness", length=300)
        br_scale.set(0)  # Varsayılan parlaklık
        br_scale.pack(pady=5)

        contrast_slider = tk.Scale(frameGreyTwo, from_=50, to=200, orient=tk.HORIZONTAL, label="Contrast", length=300)
        contrast_slider.set(100)  # Varsayılan kontrast
        contrast_slider.pack(pady=5)

        transformation_button = ttk.Button(frameGreyThree, text="Apply Grey Transformations", command=apply_grey_level_transformations)
        transformation_button.pack(pady=10)

    except Exception as e:
        error_label = tk.Label(image_frame, text="ERROR")
        error_label.pack()

    root.mainloop()

image_path = "resim.jpg"
create_form(image_path)
