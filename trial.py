import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import json
from datetime import datetime
import csv

# Download NLTK data (Run once)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')

# ==========================================
# 1. OOP CONCEPTS: CLASS & OBJECT, ENCAPSULATION
# ==========================================
class Feedback:
    """Represents a single student feedback entry (Encapsulation)."""
    def __init__(self, id, text, sentiment, source, date):
        self.__id = id                 # Private
        self.__text = text             # Private
        self.__sentiment = sentiment   # Private
        self.__source = source         # Private
        self.__date = date             # Private

    # Getters and Setters (Access Specifiers)
    def get_id(self): return self.__id
    def get_text(self): return self.__text
    def get_sentiment(self): return self.__sentiment
    def get_source(self): return self.__source
    def get_date(self): return self.__date

    def set_sentiment(self, sentiment): self.__sentiment = sentiment

    def to_dict(self):
        return {
            "id": self.__id, "text": self.__text, "sentiment": self.__sentiment,
            "source": self.__source, "date": self.__date
        }

class SentimentAnalyzer:
    """Handles NLP and Sentiment Analysis logic using VADER."""
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))

    def analyze_sentiment(self, text):
        """Analyzes text and returns Positive, Negative, or Neutral based on compound score."""
        if not text or not text.strip():
            return "Neutral"
        
        # VADER returns a dictionary: {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
        scores = self.sia.polarity_scores(text)
        compound = scores['compound']

        # Standard VADER thresholds
        if compound >= 0.05:
            return "Positive"
        elif compound <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    def extract_keywords(self, text_list, top_n=10):
        """Extracts top keywords using frequency distribution."""
        all_words = []
        for text in text_list:
            tokens = word_tokenize(text.lower())
            all_words.extend([word for word in tokens if word.isalnum() and word not in self.stop_words])
        
        freq_dist = nltk.FreqDist(all_words)
        return freq_dist.most_common(top_n)

class DataManager:
    """Handles File Handling (Read/Write) for JSON and CSV."""
    def __init__(self, json_file="feedback_data.json"):
        self.json_file = json_file

    def save_to_json(self, feedback_list):
        data = [fb.to_dict() for fb in feedback_list]
        with open(self.json_file, 'w') as f:
            json.dump(data, f, indent=4)

    def load_from_json(self):
        if os.path.exists(self.json_file):
            with open(self.json_file, 'r') as f:
                data = json.load(f)
                return [Feedback(item['id'], item['text'], item['sentiment'], item['source'], item['date']) for item in data]
        return []

    def export_to_csv(self, feedback_list):
        if not feedback_list:
            messagebox.showwarning("Export", "No data to export.")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            with open(file_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Feedback Text", "Sentiment", "Source", "Date"])
                for fb in feedback_list:
                    writer.writerow([fb.get_id(), fb.get_text(), fb.get_sentiment(), fb.get_source(), fb.get_date()])
            messagebox.showinfo("Success", "Data exported to CSV successfully!")

# ==========================================
# 2. MAIN APPLICATION & GUI (Tkinter)
# ==========================================
class FeedbackApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Feedback and Sentiment Analyzer")
        self.root.geometry("1300x800")
        self.root.configure(bg="#f4f7f6")

        self.analyzer = SentimentAnalyzer()
        self.data_manager = DataManager()
        self.feedback_data = self.data_manager.load_from_json()

        self.setup_styles()
        self.create_sidebar()
        self.create_main_content()
        self.show_dashboard() # Start on Dashboard

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Treeview", background="white", foreground="black", rowheight=30, fieldbackground="white", font=('Helvetica', 10))
        style.configure("Treeview.Heading", font=('Helvetica', 11, 'bold'), background="#e0e0e0")
        style.map('Treeview', background=[('selected', '#3498db')])

    def create_sidebar(self):
        sidebar = tk.Frame(self.root, bg="#2c3e50", width=220)
        sidebar.pack(side="left", fill="y")

        # Logo/Title
        tk.Label(sidebar, text="🎓 Feedback\nAnalyzer", bg="#2c3e50", fg="white", 
                 font=("Helvetica", 18, "bold"), justify="center").pack(pady=30)

        # Sidebar Buttons (Matching Front Page)
        buttons = [
            ("🏠 Dashboard", self.show_dashboard),
            ("➕ Add Feedback", self.show_add_feedback),
            ("📋 View Feedback", self.show_view_feedback),
            ("📊 Analyze", self.show_analyze),
            ("📈 Reports", self.show_reports),
            ("🔍 Search", self.show_search),
            ("⚙️ Settings", self.show_settings),
            ("🚪 Exit", self.root.quit)
        ]

        for text, command in buttons:
            btn = tk.Button(sidebar, text=text, command=command, bg="#34495e", fg="white", 
                            font=("Helvetica", 12), bd=0, anchor="w", padx=20, pady=12, cursor="hand2")
            btn.pack(fill="x", pady=2)
            # Hover effect
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg="#1abc9c"))
            btn.bind("<Leave>", lambda e, b=btn: b.configure(bg="#34495e"))

    def create_main_content(self):
        self.main_frame = tk.Frame(self.root, bg="#f4f7f6")
        self.main_frame.pack(side="right", fill="both", expand=True)

    def clear_main_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    # ==========================================
    # 3. VIEWS / PAGES
    # ==========================================
    
    # --- DASHBOARD ---
    def show_dashboard(self):
        self.clear_main_frame()
        
        # Header
        header_frame = tk.Frame(self.main_frame, bg="#f4f7f6")
        header_frame.pack(fill="x", padx=20, pady=15)
        tk.Label(header_frame, text="Dashboard", font=("Helvetica", 26, "bold"), bg="#f4f7f6", fg="#2c3e50").pack(side="left")

        # Stats Cards
        stats_frame = tk.Frame(self.main_frame, bg="#f4f7f6")
        stats_frame.pack(fill="x", padx=20)

        total = len(self.feedback_data)
        pos = sum(1 for f in self.feedback_data if f.get_sentiment() == "Positive")
        neg = sum(1 for f in self.feedback_data if f.get_sentiment() == "Negative")
        neu = sum(1 for f in self.feedback_data if f.get_sentiment() == "Neutral")

        self.create_stat_card(stats_frame, "Total Feedback", total, "#3498db")
        self.create_stat_card(stats_frame, "Positive", f"{pos} ({int(pos/total*100) if total>0 else 0}%)", "#2ecc71")
        self.create_stat_card(stats_frame, "Negative", f"{neg} ({int(neg/total*100) if total>0 else 0}%)", "#e74c3c")
        self.create_stat_card(stats_frame, "Neutral", f"{neu} ({int(neu/total*100) if total>0 else 0}%)", "#f1c40f")

        # Charts Area
        chart_frame = tk.Frame(self.main_frame, bg="#f4f7f6")
        chart_frame.pack(fill="both", expand=True, padx=20, pady=10)

        if total > 0:
            # Pie Chart
            fig1, ax1 = plt.subplots(figsize=(4, 3.5))
            labels = ['Positive', 'Negative', 'Neutral']
            sizes = [pos, neg, neu]
            colors = ['#2ecc71', '#e74c3c', '#f1c40f']
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90, wedgeprops=dict(width=0.4))
            ax1.set_title("Sentiment Distribution", fontweight='bold')
            self.embed_chart(fig1, chart_frame, 0, 0)

            # Top Keywords Bar Chart
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            keywords = self.analyzer.extract_keywords([f.get_text() for f in self.feedback_data], top_n=6)
            if keywords:
                words, counts = zip(*keywords)
                ax2.barh(words, counts, color='#9b59b6')
                ax2.set_title("Top Keywords", fontweight='bold')
                ax2.invert_yaxis() # Top keyword at top
            self.embed_chart(fig2, chart_frame, 0, 1)
        else:
            tk.Label(chart_frame, text="No data available. Please add feedback.", bg="#f4f7f6", font=("Helvetica", 14)).pack(pady=50)

        # Recent Feedback Table
        table_frame = tk.Frame(self.main_frame, bg="white", relief="groove", bd=1)
        table_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        tk.Label(table_frame, text="Recent Feedback", font=("Helvetica", 14, "bold"), bg="white").pack(anchor="w", padx=10, pady=5)
        
        columns = ("ID", "Feedback", "Sentiment", "Source")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=6)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        
        # Show last 5 feedbacks
        for fb in self.feedback_data[-5:]:
            tree.insert("", "end", values=(fb.get_id(), fb.get_text()[:80] + "...", fb.get_sentiment(), fb.get_source()))
        
        tree.pack(fill="x", padx=10, pady=5)

    def create_stat_card(self, parent, title, value, color):
        card = tk.Frame(parent, bg=color, bd=0, relief="flat", padx=15, pady=15)
        card.pack(side="left", fill="x", expand=True, padx=5)
        
        tk.Label(card, text=title, bg=color, fg="white", font=("Helvetica", 11)).pack(anchor="w")
        tk.Label(card, text=str(value), bg=color, fg="white", font=("Helvetica", 22, "bold")).pack(anchor="w")

    def embed_chart(self, fig, parent, row, col):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        parent.grid_columnconfigure(col, weight=1)

    # --- ADD FEEDBACK ---
    def show_add_feedback(self):
        self.clear_main_frame()
        tk.Label(self.main_frame, text="Add New Feedback", font=("Helvetica", 26, "bold"), bg="#f4f7f6", fg="#2c3e50").pack(anchor="w", padx=20, pady=20)

        form_frame = tk.Frame(self.main_frame, bg="white", padx=30, pady=30, relief="groove", bd=1)
        form_frame.pack(padx=40, pady=10, fill="x")

        tk.Label(form_frame, text="Feedback Text:", bg="white", font=("Helvetica", 12, "bold")).pack(anchor="w")
        self.text_entry = tk.Text(form_frame, height=6, width=70, font=("Helvetica", 11), bd=2, relief="solid")
        self.text_entry.pack(pady=10)

        tk.Label(form_frame, text="Source (e.g., Student, Parent):", bg="white", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(10,0))
        self.source_entry = tk.Entry(form_frame, width=40, font=("Helvetica", 11), bd=2, relief="solid")
        self.source_entry.pack(pady=10)

        submit_btn = tk.Button(form_frame, text="Submit & Analyze", command=self.submit_feedback, 
                               bg="#3498db", fg="white", font=("Helvetica", 12, "bold"), padx=30, pady=8, cursor="hand2")
        submit_btn.pack(pady=20)

    def submit_feedback(self):
        text = self.text_entry.get("1.0", tk.END).strip()
        source = self.source_entry.get().strip() or "Student"

        if not text:
            messagebox.showwarning("Input Error", "Please enter feedback text.")
            return

        # Analyze Sentiment using VADER
        sentiment = self.analyzer.analyze_sentiment(text)
        
        # Create Object
        fb_id = len(self.feedback_data) + 1
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        new_feedback = Feedback(fb_id, text, sentiment, source, date_str)
        
        self.feedback_data.append(new_feedback)
        self.data_manager.save_to_json(self.feedback_data)
        
        # Color code the message box based on sentiment
        if sentiment == "Positive":
            messagebox.showinfo("Success", f"Feedback added!\nDetected Sentiment: {sentiment} 🟢")
        elif sentiment == "Negative":
            messagebox.showinfo("Success", f"Feedback added!\nDetected Sentiment: {sentiment} 🔴")
        else:
            messagebox.showinfo("Success", f"Feedback added!\nDetected Sentiment: {sentiment} 🟡")
            
        self.show_dashboard()

    # --- VIEW FEEDBACK ---
    def show_view_feedback(self):
        self.clear_main_frame()
        tk.Label(self.main_frame, text="All Feedback Records", font=("Helvetica", 26, "bold"), bg="#f4f7f6", fg="#2c3e50").pack(anchor="w", padx=20, pady=20)

        columns = ("ID", "Feedback", "Sentiment", "Source", "Date")
        tree = ttk.Treeview(self.main_frame, columns=columns, show="headings")
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        
        for fb in self.feedback_data:
            tree.insert("", "end", values=(fb.get_id(), fb.get_text(), fb.get_sentiment(), fb.get_source(), fb.get_date()))
        
        tree.pack(fill="both", expand=True, padx=20, pady=10)

        btn_frame = tk.Frame(self.main_frame, bg="#f4f7f6")
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Delete Selected", command=lambda: self.delete_feedback(tree),
                  bg="#e74c3c", fg="white", font=("Helvetica", 11, "bold"), padx=20, pady=5).pack(side="left", padx=10)
        tk.Button(btn_frame, text="Export to CSV", command=lambda: self.data_manager.export_to_csv(self.feedback_data),
                  bg="#2ecc71", fg="white", font=("Helvetica", 11, "bold"), padx=20, pady=5).pack(side="left", padx=10)

    def delete_feedback(self, tree):
        selected = tree.selection()
        if not selected:
            messagebox.showwarning("Select", "Please select a row to delete.")
            return
        
        if messagebox.askyesno("Confirm", "Are you sure you want to delete this feedback?"):
            for item in selected:
                values = tree.item(item, 'values')
                fb_id = int(values[0])
                self.feedback_data = [f for f in self.feedback_data if f.get_id() != fb_id]
                tree.delete(item)
            
            self.data_manager.save_to_json(self.feedback_data)
            messagebox.showinfo("Deleted", "Feedback deleted successfully.")

    # --- ANALYZE ---
    def show_analyze(self):
        self.clear_main_frame()
        tk.Label(self.main_frame, text="Deep Analysis", font=("Helvetica", 26, "bold"), bg="#f4f7f6", fg="#2c3e50").pack(anchor="w", padx=20, pady=20)
        
        if not self.feedback_data:
            tk.Label(self.main_frame, text="No data to analyze.", bg="#f4f7f6", font=("Helvetica", 14)).pack(pady=20)
            return

        # Trend Chart (Simulated over time based on ID)
        fig, ax = plt.subplots(figsize=(8, 4))
        dates = [fb.get_date()[:10] for fb in self.feedback_data]
        sentiments = [1 if fb.get_sentiment() == "Positive" else (-1 if fb.get_sentiment() == "Negative" else 0) for fb in self.feedback_data]
        
        ax.plot(range(len(sentiments)), sentiments, marker='o', linestyle='-', color='#3498db')
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(['Negative', 'Neutral', 'Positive'])
        ax.set_title("Sentiment Trend Over Time", fontweight='bold')
        ax.set_xlabel("Feedback Entry Number")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        canvas = FigureCanvasTkAgg(fig, master=self.main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=10)

    # --- REPORTS ---
    def show_reports(self):
        self.clear_main_frame()
        tk.Label(self.main_frame, text="System Reports", font=("Helvetica", 26, "bold"), bg="#f4f7f6", fg="#2c3e50").pack(anchor="w", padx=20, pady=20)
        
        report_text = tk.Text(self.main_frame, height=15, width=80, font=("Courier", 11))
        report_text.pack(padx=20, pady=10)
        
        total = len(self.feedback_data)
        if total == 0:
            report_text.insert(tk.END, "No data available.")
        else:
            pos = sum(1 for f in self.feedback_data if f.get_sentiment() == "Positive")
            neg = sum(1 for f in self.feedback_data if f.get_sentiment() == "Negative")
            neu = sum(1 for f in self.feedback_data if f.get_sentiment() == "Neutral")
            
            report = f"--- STUDENT FEEDBACK SUMMARY REPORT ---\n"
            report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            report += f"Total Feedback Analyzed: {total}\n"
            report += f"Positive: {pos} ({pos/total*100:.1f}%)\n"
            report += f"Negative: {neg} ({neg/total*100:.1f}%)\n"
            report += f"Neutral:  {neu} ({neu/total*100:.1f}%)\n\n"
            report += "Top Keywords Identified:\n"
            keywords = self.analyzer.extract_keywords([f.get_text() for f in self.feedback_data], top_n=5)
            for word, count in keywords:
                report += f" - {word}: {count} times\n"
                
            report_text.insert(tk.END, report)
        report_text.config(state="disabled")

    # --- SEARCH ---
    def show_search(self):
        self.clear_main_frame()
        tk.Label(self.main_frame, text="Search Feedback", font=("Helvetica", 26, "bold"), bg="#f4f7f6", fg="#2c3e50").pack(anchor="w", padx=20, pady=20)

        search_frame = tk.Frame(self.main_frame, bg="#f4f7f6")
        search_frame.pack(fill="x", padx=20)
        
        tk.Label(search_frame, text="Enter keyword:", bg="#f4f7f6", font=("Helvetica", 12)).pack(side="left")
        self.search_entry = tk.Entry(search_frame, width=40, font=("Helvetica", 12))
        self.search_entry.pack(side="left", padx=10)
        
        tk.Button(search_frame, text="Search", command=self.perform_search, bg="#3498db", fg="white", font=("Helvetica", 10, "bold")).pack(side="left")

        self.search_results_frame = tk.Frame(self.main_frame, bg="#f4f7f6")
        self.search_results_frame.pack(fill="both", expand=True, padx=20, pady=20)

    def perform_search(self):
        query = self.search_entry.get().lower()
        for widget in self.search_results_frame.winfo_children():
            widget.destroy()
            
        results = [fb for fb in self.feedback_data if query in fb.get_text().lower()]
        
        if not results:
            tk.Label(self.search_results_frame, text="No matching feedback found.", bg="#f4f7f6", font=("Helvetica", 12)).pack()
            return

        columns = ("ID", "Feedback", "Sentiment", "Source")
        tree = ttk.Treeview(self.search_results_frame, columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        
        for fb in results:
            tree.insert("", "end", values=(fb.get_id(), fb.get_text(), fb.get_sentiment(), fb.get_source()))
        tree.pack(fill="both", expand=True)

    # --- SETTINGS ---
    def show_settings(self):
        self.clear_main_frame()
        tk.Label(self.main_frame, text="Settings", font=("Helvetica", 26, "bold"), bg="#f4f7f6", fg="#2c3e50").pack(anchor="w", padx=20, pady=20)
        
        settings_frame = tk.Frame(self.main_frame, bg="white", padx=30, pady=30, relief="groove", bd=1)
        settings_frame.pack(padx=40, pady=10, fill="x")
        
        tk.Label(settings_frame, text="Data Management", bg="white", font=("Helvetica", 14, "bold")).pack(anchor="w", pady=(0,10))
        
        tk.Button(settings_frame, text="Clear All Data", command=self.clear_all_data, 
                  bg="#e74c3c", fg="white", font=("Helvetica", 11, "bold"), padx=20, pady=5).pack(anchor="w", pady=5)
        
        tk.Label(settings_frame, text="About", bg="white", font=("Helvetica", 14, "bold")).pack(anchor="w", pady=(20,10))
        tk.Label(settings_frame, text="Student Feedback and Sentiment Analyzer\nVersion 1.0\nPowered by Python, Tkinter, and NLTK.", 
                 bg="white", font=("Helvetica", 11), justify="left").pack(anchor="w")

    def clear_all_data(self):
        if messagebox.askyesno("Warning", "This will delete ALL feedback data permanently. Are you sure?"):
            self.feedback_data = []
            self.data_manager.save_to_json(self.feedback_data)
            messagebox.showinfo("Cleared", "All data has been cleared.")
            self.show_dashboard()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    app = FeedbackApp(root)
    root.mainloop()