import os
import warnings
# Suppress TensorFlow deprecation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', module='tensorflow')
import time
import csv
import warnings
import numpy as np
import customtkinter
import tkinter as tk
from tkinter import END, messagebox
from tkinter.filedialog import askopenfilenames
from docx import Document
from PyPDF2 import PdfReader
from fpdf import FPDF
from difflib import SequenceMatcher
from PIL import Image, ImageDraw, ImageFont, ImageTk
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from joblib import Parallel, delayed
from collections import defaultdict

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# Initialize Sentence Transformer model
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# Create necessary folders
for folder in ["Pending", "Reports", "Screenshots"]:
    os.makedirs(folder, exist_ok=True)

# File conversion functions (unchanged)
def convert_docx_to_txt(docx_file, txt_file):
    document = Document(docx_file)
    with open(txt_file, "w", encoding="utf-8") as txt:
        for paragraph in document.paragraphs:
            txt.write(paragraph.text + "\n")

def convert_pdf_to_txt(pdf_file, txt_file):
    reader = PdfReader(pdf_file)
    with open(txt_file, "w", encoding="utf-8") as txt:
        for page in reader.pages:
            txt.write(page.extract_text() + "\n")

def convert_to_txt(input_file):
    base_name = os.path.basename(os.path.splitext(input_file)[0])
    output_file = os.path.join("Pending", f"{base_name}.txt")
    ext = os.path.splitext(input_file)[-1].lower()
    try:
        if ext == ".docx":
            convert_docx_to_txt(input_file, output_file)
        elif ext == ".pdf":
            convert_pdf_to_txt(input_file, output_file)
        elif ext == ".txt":
            with open(input_file, "r", encoding="utf-8") as f_in:
                with open(output_file, "w", encoding="utf-8") as f_out:
                    f_out.write(f_in.read())
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        print(f"Error converting file: {e}")

# Global variables
uploaded_files = []
plagiarism_results = []

def upload_files():
    filepaths = askopenfilenames(filetypes=[("All files", "*.*")])
    if not filepaths:
        return

    global uploaded_files
    uploaded_files = []

    for filepath in filepaths:
        base_name = os.path.basename(filepath)
        txt_path = os.path.join("Pending", f"{os.path.splitext(base_name)[0]}.txt")
        convert_to_txt(filepath)
        uploaded_files.append(txt_path)

    messagebox.showinfo("Upload Complete", "Files uploaded and converted successfully.")

def extract_copied_texts(file1, file2, min_similarity=0.7):
    with open(file1, encoding='utf-8') as f1, open(file2, encoding='utf-8') as f2:
        text1 = f1.read()
        text2 = f2.read()

    matcher = SequenceMatcher(None, text1, text2)
    copied_texts = []

    # Extract matching blocks with sufficient size and similarity
    for match in matcher.get_matching_blocks():
        if match.size > 10:  # Minimum length for a meaningful match
            copied_text = text1[match.a:match.a + match.size].strip()
            if copied_text:
                # Calculate similarity for the matched block
                block_similarity = matcher.real_quick_ratio()  # Approximate similarity
                if block_similarity >= min_similarity:
                    copied_texts.append({
                        "source_document_1": os.path.basename(file1),
                        "source_document_2": os.path.basename(file2),
                        "copied_text": copied_text,
                        "similarity": block_similarity
                    })

    # If no copied texts were found but similarity is above threshold, return placeholder
    if not copied_texts:
        # Re-calculate overall similarity using cosine similarity
        embeds = MODEL.encode([text1, text2])
        overall_similarity = cosine_similarity([embeds[0]], [embeds[1]])[0][0]
        if overall_similarity >= min_similarity:
            copied_texts.append({
                "source_document_1": os.path.basename(file1),
                "source_document_2": os.path.basename(file2),
                "copied_text": "(No specific copied text detected, but overall similarity is high.)",
                "similarity": overall_similarity
            })

    return copied_texts

def check_uploaded_files_plagiarism():
    global uploaded_files, plagiarism_results
    if not uploaded_files:
        messagebox.showwarning("No Files", "No files have been uploaded.")
        return

    try:
        corrupt_folder = os.path.join("Pending", "corrupt documents")
        os.makedirs(corrupt_folder, exist_ok=True)

        valid_files = []
        corrupted_files = []
        file_contents = []

        for file in uploaded_files:
            try:
                with open(file, encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        file_name = os.path.basename(file)
                        corrupt_file_path = os.path.join(corrupt_folder, file_name)
                        os.rename(file, corrupt_file_path)
                        corrupted_files.append(file_name)
                        messagebox.showwarning("Empty File", f"The file {file_name} is empty and has been moved to 'corrupt documents'.")
                        continue
                    file_contents.append(content)
                    valid_files.append(file)
            except Exception as e:
                file_name = os.path.basename(file)
                corrupt_file_path = os.path.join(corrupt_folder, file_name)
                os.rename(file, corrupt_file_path)
                corrupted_files.append(file_name)
                messagebox.showwarning("File Access Error", f"The file {file_name} could not be accessed and has been moved to 'corrupt documents'. Error: {e}")

        if not file_contents:
            messagebox.showerror("Empty Files", "All uploaded files are empty, corrupted, or contain no meaningful content.")
            return

        if len(valid_files) < len(uploaded_files):
            continue_check = messagebox.askyesno("Continue?", "Some files were corrupted or empty and moved to 'corrupt documents'. Do you want to continue with the remaining files?")
            if not continue_check:
                return

        # Optimized processing starts here
        start_time = time.time()
        
        # Generate embeddings
        embeddings = MODEL.encode(file_contents)
        print(f"Embedding generation time: {time.time() - start_time:.2f}s")

        # Cluster optimization
        def optimal_cluster_count(embeddings, max_clusters=5):
            max_clusters = min(max_clusters, len(embeddings)-1)
            best_score = -1
            optimal = 2
            
            for n in range(2, max_clusters+1):
                kmeans = MiniBatchKMeans(n_clusters=n, random_state=42, batch_size=100)
                kmeans.fit(embeddings)
                score = silhouette_score(embeddings, kmeans.labels_)
                if score > best_score:
                    best_score = score
                    optimal = n
            return optimal

        cluster_start = time.time()
        optimal_clusters = optimal_cluster_count(embeddings)
        kmeans = MiniBatchKMeans(n_clusters=optimal_clusters, random_state=42, batch_size=100)
        clusters = kmeans.fit_predict(embeddings)
        print(f"Clustering time: {time.time() - cluster_start:.2f}s")

        # Group documents by cluster
        cluster_groups = defaultdict(list)
        for idx, cluster_id in enumerate(clusters):
            cluster_groups[cluster_id].append((valid_files[idx], embeddings[idx]))

        # Parallel comparison
        compare_start = time.time()
        
        def cluster_comparison(cluster_docs):
            filenames = [doc[0] for doc in cluster_docs]
            embeds = np.array([doc[1] for doc in cluster_docs])
            sim_matrix = cosine_similarity(embeds)
            results = []
            
            for i in range(sim_matrix.shape[0]):
                for j in range(i+1, sim_matrix.shape[1]):
                    if sim_matrix[i,j] >= 0.7:
                        pair = sorted([os.path.basename(filenames[i]), os.path.basename(filenames[j])])
                        results.append({
                            "Assignment 1": pair[0],
                            "Assignment 2": pair[1],
                            "Similarity Score": f"{sim_matrix[i,j]:.2f}",
                            "Plagiarism Status": "Flagged" if sim_matrix[i,j] >= 0.7 else "Clean"
                        })
            return results

        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(cluster_comparison)(cluster) 
            for cluster in cluster_groups.values()
        )
        
        plagiarism_results = [item for sublist in results for item in sublist]
        print(f"Comparison time: {time.time() - compare_start:.2f}s")
        print(f"Total processing time: {time.time() - start_time:.2f}s")

        if corrupted_files:
            messagebox.showinfo(
                "Plagiarism Check Complete",
                f"Plagiarism check completed successfully.\n\n"
                f"The following files were corrupted or empty and moved to 'corrupt documents':\n"
                f"{', '.join(corrupted_files)}"
            )
        else:
            messagebox.showinfo("Plagiarism Check Complete", "Plagiarism check completed successfully.")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during the plagiarism check: {e}")

def show_copied_texts():
    if not plagiarism_results:
        messagebox.showerror("No Results", "No plagiarism results to display.")
        return

    result_window = customtkinter.CTkToplevel()
    result_window.title("Plagiarism Results - Command Line Format")
    result_window.geometry("1200x800")
    result_window.state("zoomed")
    main_frame = customtkinter.CTkFrame(result_window)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)
    text_widget = tk.Text(main_frame, wrap="word", font=("Consolas", 12))
    scrollbar = customtkinter.CTkScrollbar(main_frame, command=text_widget.yview)
    text_widget.configure(yscrollcommand=scrollbar.set)
    text_widget.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")
    main_frame.grid_rowconfigure(0, weight=1)
    main_frame.grid_columnconfigure(0, weight=1)

    # Header for the results
    text_widget.insert("end", "\nPlagiarism Results:\n")
    text_widget.insert("end", "{:<10} {:<30} {:<30} {:<20}\n".format(
        "ID", "Source Document", "Copied Document", "Similarity Score"
    ))
    text_widget.insert("end", "-" * 100 + "\n")

    id_counter = 1
    shown_copied_texts = set()  # Track already shown copied texts

    for result in plagiarism_results:
        if result["Plagiarism Status"] == "Flagged":
            source_doc = result["Assignment 1"]
            copied_doc = result["Assignment 2"]
            sim_score = float(result["Similarity Score"])

            # Display flagged pair information
            text_widget.insert("end", "{:<10} {:<30} {:<30} {:.2f}\n".format(
                id_counter, source_doc, copied_doc, sim_score
            ))

            file1 = os.path.join("Pending", source_doc)
            file2 = os.path.join("Pending", copied_doc)
            copied_texts = extract_copied_texts(file1, file2)

            if copied_texts:
                for copied_text in copied_texts:
                    # Ensure the copied text is not already shown
                    if copied_text['copied_text'] not in shown_copied_texts:
                        text_widget.insert("end", f"\nHighlighted Copied Text:\n{copied_text['copied_text']}\n")
                        shown_copied_texts.add(copied_text['copied_text'])  # Mark as shown
            else:
                text_widget.insert("end", "\nNo specific copied texts detected, but similarity is above threshold.\n")

            text_widget.insert("end", "-" * 100 + "\n\n")
            id_counter += 1

    # Configure text widget appearance
    text_color = "#FFFFFF" if customtkinter.get_appearance_mode() == "Dark" else "#000000"
    text_widget.configure(fg=text_color, bg=main_frame.cget("fg_color"))
    text_widget.configure(state="disabled")

    # Close button
    close_btn = customtkinter.CTkButton(
        main_frame,
        text="Close Window",
        command=result_window.destroy,
        fg_color="#4CAF50",
        hover_color="#45a049"
    )
    close_btn.grid(row=1, column=0, pady=10)

def export_report():
    if not plagiarism_results:
        messagebox.showerror("No Results", "No plagiarism results to export.")
        return

    flagged_results = [result for result in plagiarism_results if result["Plagiarism Status"] == "Flagged"]

    if not flagged_results:
        messagebox.showinfo("No Flagged Documents", "No flagged documents to export.")
        return

    simplified_results = [
        {
            "Source Document 1": result["Assignment 1"],
            "Source Document 2": result["Assignment 2"],
            "Similarity Score": result["Similarity Score"],
            "Plagiarism Status": result["Plagiarism Status"]
        }
        for result in flagged_results
    ]

    csv_filename = os.path.join("Reports", "plagiarism_report.csv")
    keys = simplified_results[0].keys()
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(simplified_results)

    pdf_filename = os.path.join("Reports", "plagiarism_report.pdf")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Plagiarism Report", ln=True, align='C')
    pdf.ln(10)

    for result in simplified_results:
        pdf.multi_cell(0, 10, txt=f"Source Document 1: {result['Source Document 1']}\n"
                                  f"Source Document 2: {result['Source Document 2']}\n"
                                  f"Similarity Score: {result['Similarity Score']}\n"
                                  f"Plagiarism Status: {result['Plagiarism Status']}\n")
        pdf.ln(5)

    pdf.output(pdf_filename)
    messagebox.showinfo("Report Generated", f"Report saved successfully in Reports folder.\nCSV: {csv_filename}\nPDF: {pdf_filename}")

def open_report():
    report_choice = messagebox.askquestion("Open Report", "Would you like to open the report in PDF format? Click 'Yes' for PDF and 'No' for CSV.")
    if report_choice == 'yes':
        os.system(f'start Reports/plagiarism_report.pdf')
    else:
        os.system(f'start Reports/plagiarism_report.csv')

def reset_application():
    global uploaded_files, plagiarism_results
    uploaded_files = []
    plagiarism_results = []

    for file in os.listdir("Pending"):
        file_path = os.path.join("Pending", file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    for file in os.listdir("Screenshots"):
        file_path = os.path.join("Screenshots", file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    messagebox.showinfo("Reset Complete", "Application has been reset to its initial state.")

def toggle_appearance_mode():
    current_mode = customtkinter.get_appearance_mode()
    if current_mode == "Dark":
        customtkinter.set_appearance_mode("Light")
        btn_toggle_mode.configure(image=moon_icon)
    else:
        customtkinter.set_appearance_mode("Dark")
        btn_toggle_mode.configure(image=sun_icon)
    update_widget_colors()

def update_widget_colors():
    current_mode = customtkinter.get_appearance_mode()
    if current_mode == "Light":
        window.configure(fg_color="#F5F5F5")
        welcome_frm.configure(fg_color="#F5F5F5")
        welcome_lbl.configure(text_color="#333333")
        button_frm.configure(fg_color="#F5F5F5")
        for button in [btn_upload, btn_check, btn_report, btn_open_report, btn_reset, btn_show_copied_texts]:
            button.configure(fg_color="#4CAF50", hover_color="#45a049", text_color="#FFFFFF")
        btn_reset.configure(fg_color="#FF0000", hover_color="#CC0000")
        btn_toggle_mode.configure(fg_color="#F5F5F5", hover_color="#F5F5F5", text_color="#000000")
    else:
        window.configure(fg_color="#2E2E2E")
        welcome_frm.configure(fg_color="#2E2E2E")
        welcome_lbl.configure(text_color="yellow")
        button_frm.configure(fg_color="#2E2E2E")
        for button in [btn_upload, btn_check, btn_report, btn_open_report, btn_reset, btn_show_copied_texts]:
            button.configure(fg_color="#4CAF50", hover_color="#45a049", text_color="#FFFFFF")
        btn_reset.configure(fg_color="#FF0000", hover_color="#CC0000")
        btn_toggle_mode.configure(fg_color="#2E2E2E", hover_color="#2E2E2E", text_color="#FFFFFF")

# GUI implementation
window = customtkinter.CTk()
customtkinter.set_appearance_mode("dark")
window.title("Plagiarism Checker System")
window.state("zoomed")

# Load icons
sun_icon = ImageTk.PhotoImage(Image.open("sun.png").resize((32, 32))) if os.path.exists("sun.png") else None
moon_icon = ImageTk.PhotoImage(Image.open("moon.png").resize((32, 32))) if os.path.exists("moon.png") else None

# GUI components
welcome_frm = customtkinter.CTkFrame(window)
welcome_msg_variable = tk.StringVar(welcome_frm, "Welcome to the state of the art Plagiarism Checker System")
welcome_lbl = customtkinter.CTkLabel(welcome_frm, textvariable=welcome_msg_variable,
                                     height=100, corner_radius=20, 
                                     text_color="yellow", font=("Comic Sans MS bold", 30))
welcome_lbl.grid(row=0, column=0, padx=200, pady=(20, 0), sticky="nsew")

button_frm = customtkinter.CTkFrame(window)
btn_upload = customtkinter.CTkButton(button_frm, corner_radius=30, hover_color="Green", text="UPLOAD FILES", command=upload_files)
btn_upload.grid(row=0, column=0, padx=(180, 10), pady=10)

btn_check = customtkinter.CTkButton(button_frm, corner_radius=30, hover_color="Green", text="CHECK PLAGIARISM", command=check_uploaded_files_plagiarism)
btn_check.grid(row=0, column=1, padx=10, pady=10)

btn_report = customtkinter.CTkButton(button_frm, corner_radius=30, hover_color="Green", text="EXPORT REPORT", command=export_report)
btn_report.grid(row=0, column=2, padx=10, pady=10)

btn_open_report = customtkinter.CTkButton(button_frm, corner_radius=30, hover_color="Green", text="OPEN REPORT", command=open_report)
btn_open_report.grid(row=0, column=3, padx=10, pady=10)

btn_reset = customtkinter.CTkButton(button_frm, corner_radius=30, hover_color="Red", text="RESET", command=reset_application)
btn_reset.grid(row=0, column=4, padx=10, pady=10)

btn_show_copied_texts = customtkinter.CTkButton(button_frm, corner_radius=30, hover_color="Green", text="SHOW COPIED TEXTS", command=show_copied_texts)
btn_show_copied_texts.grid(row=0, column=5, padx=10, pady=10)

top_right_frame = customtkinter.CTkFrame(window)
top_right_frame.grid(row=0, column=1, padx=20, pady=20, sticky="ne")

btn_toggle_mode = customtkinter.CTkButton(top_right_frame, text="", image=sun_icon if sun_icon else None, 
                                         command=toggle_appearance_mode, fg_color="#2E2E2E", hover_color="#2E2E2E", 
                                         width=40, height=40)
btn_toggle_mode.grid(row=0, column=0)

welcome_frm.grid(row=0, column=0, padx=20, pady=(50, 10), sticky="nsew")
button_frm.grid(row=1, column=0, padx=20, pady=(10, 50), sticky="nsew")

window.grid_rowconfigure(0, weight=0)
window.grid_rowconfigure(0, weight=0)
window.grid_columnconfigure(0, weight=1)

update_widget_colors()

window.mainloop()