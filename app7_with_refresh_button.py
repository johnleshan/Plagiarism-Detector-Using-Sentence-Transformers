import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import customtkinter
import tkinter as tk
from tkinter import END, messagebox
from tkinter.filedialog import askopenfilenames
from docx import Document
from PyPDF2 import PdfReader
from fpdf import FPDF

# Create necessary folders
if not os.path.exists("Pending"):
    os.makedirs("Pending")
if not os.path.exists("Reports"):
    os.makedirs("Reports")

# File conversion functions

def convert_docx_to_txt(docx_file, txt_file):
    """Convert .docx to .txt using python-docx."""
    document = Document(docx_file)
    with open(txt_file, "w", encoding="utf-8") as txt:
        for paragraph in document.paragraphs:
            txt.write(paragraph.text + "\n")

def convert_pdf_to_txt(pdf_file, txt_file):
    """Convert .pdf to .txt using PyPDF2."""
    reader = PdfReader(pdf_file)
    with open(txt_file, "w", encoding="utf-8") as txt:
        for page in reader.pages:
            txt.write(page.extract_text() + "\n")

def convert_to_txt(input_file):
    """Determine file type and convert to .txt, saving in the Pending folder."""
    base_name = os.path.basename(os.path.splitext(input_file)[0])  # Get the file name without extension
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
        print(f"Conversion successful: {input_file} -> {output_file}")
    except Exception as e:
        print(f"Error converting file: {e}")

# Global list to store paths of uploaded files
uploaded_files = []
plagiarism_results = []

# Function to upload multiple files
def upload_files():
    """Upload multiple files and convert them to .txt in the Pending folder."""
    filepaths = askopenfilenames(
        filetypes=[
            ("All files", "*.*")
        ]
    )
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

# Function to perform plagiarism only on uploaded files
def check_uploaded_files_plagiarism():
    """Perform plagiarism check on the uploaded .txt files in the Pending folder."""
    if not uploaded_files:
        messagebox.showwarning("No Files", "No files have been uploaded.")
        return

    notes = [open(file, encoding='utf-8').read().strip() for file in uploaded_files]
    if not notes or all(len(note.split()) == 0 for note in notes):
        messagebox.showerror("Empty Files", "Uploaded files are empty or contain no meaningful content.")
        return

    vectors = TfidfVectorizer().fit_transform(notes).toarray()
    s_vectors = list(zip(uploaded_files, vectors))
    global plagiarism_results
    plagiarism_results = []

    for student_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_a, text_vector_a))
        del new_vectors[current_index]
        for student_b, text_vector_b in new_vectors:
            sim_score = cosine_similarity([text_vector_a, text_vector_b])[0][1]
            student_pair = sorted((os.path.basename(student_a), os.path.basename(student_b)))
            score = (student_pair[0], student_pair[1], sim_score)
            plagiarism_results.append({
                "Assignment 1": student_pair[0],
                "Assignment 2": student_pair[1],
                "Similarity Score": f"{sim_score:.2f}",
                "Plagiarism Status": "Flagged" if sim_score >= 0.7 else "Clean"
            })

    messagebox.showinfo("Plagiarism Check Complete", "Plagiarism check completed successfully.")

# Function to export the plagiarism report
def export_report():
    """Generate and export plagiarism report to CSV and PDF."""
    if not plagiarism_results:
        messagebox.showerror("No Results", "No plagiarism results to export.")
        return

    # Export to CSV
    csv_filename = os.path.join("Reports", "plagiarism_report.csv")
    keys = plagiarism_results[0].keys()  # Assuming all reports have the same keys
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(plagiarism_results)

    # Export to PDF
    pdf_filename = os.path.join("Reports", "plagiarism_report.pdf")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Plagiarism Report", ln=True, align='C')
    pdf.ln(10)

    for result in plagiarism_results:
        pdf.multi_cell(0, 10, txt=f"{result}")
        pdf.ln(2)

    pdf.output(pdf_filename)
    messagebox.showinfo("Report Generated", f"Report saved successfully in Reports folder.\nCSV: {csv_filename}\nPDF: {pdf_filename}")

# Function to open the generated report
def open_report():
    """Allow the user to open the report in either CSV or PDF format."""
    report_choice = messagebox.askquestion("Open Report", "Would you like to open the report in PDF format? Click 'Yes' for PDF and 'No' for CSV.")
    if report_choice == 'yes':
        os.system(f'start Reports/plagiarism_report.pdf')
    else:
        os.system(f'start Reports/plagiarism_report.csv')

# Function to reset the application
def reset_application():
    """Clear all data and reset the application to its initial state."""
    global uploaded_files, plagiarism_results
    uploaded_files = []
    plagiarism_results = []

    # Clear Pending folder
    for file in os.listdir("Pending"):
        file_path = os.path.join("Pending", file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    messagebox.showinfo("Reset Complete", "Application has been reset to its initial state.")

# GUI IMPLEMENTED USING CUSTOMTKINTER
window = customtkinter.CTk()
customtkinter.set_appearance_mode("dark")  # Always open the app in dark mode
window.title("Plagiarism Checker System")

# FRONT END DESIGN
# 1st Frame containing the welcome and intro msg label widgets.
welcome_frm = customtkinter.CTkFrame(window)
welcome_msg = "Welcome to the state of the art Plagiarism Checker System"
welcome_msg_variable = tk.StringVar(welcome_frm, welcome_msg)
welcome_lbl = customtkinter.CTkLabel(welcome_frm, textvariable=welcome_msg_variable,
                                     height=100, corner_radius=20, 
                                     text_color="yellow", font=("Comic Sans MS bold", 30))
welcome_lbl.grid(row=0, column=0, padx=20, sticky="n")

# 3rd Frame contains the buttons to help with the functionality of the whole app.
button_frm = customtkinter.CTkFrame(window)
btn_upload = customtkinter.CTkButton(
    button_frm, corner_radius=30, hover_color="Green", text="UPLOAD FILES", command=upload_files
)
btn_upload.grid(row=0, column=0, padx=(170, 0), sticky="nwne")

btn_check = customtkinter.CTkButton(
    button_frm, corner_radius=30, hover_color="Green", text="CHECK PLAGIARISM", command=check_uploaded_files_plagiarism
)
btn_check.grid(row=0, column=1, padx=(30, 0), sticky="nwne")

btn_report = customtkinter.CTkButton(
    button_frm, corner_radius=30, hover_color="Green", text="EXPORT REPORT", command=export_report
)
btn_report.grid(row=0, column=2, padx=(30, 0), sticky="nwne")

btn_open_report = customtkinter.CTkButton(
    button_frm, corner_radius=30, hover_color="Green", text="OPEN REPORT", command=open_report
)
btn_open_report.grid(row=0, column=3, padx=(30, 0), sticky="nwne")

btn_reset = customtkinter.CTkButton(
    button_frm, corner_radius=30, hover_color="Red", text="RESET", command=reset_application
)
btn_reset.grid(row=0, column=4, padx=(30, 170), sticky="nwne")

welcome_frm.grid(row=0, column=0, padx=(50, 40), pady=(155, 30), sticky="se")
button_frm.grid(row=0, column=0, padx=(50, 40), pady=(0, 0), sticky="se")

window.mainloop()