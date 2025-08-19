"""
AI-Powered Study Helper (Tkinter)
Day 30/30 Project – by Prerna Gupta

Features
- Paste text or load a PDF, then generate a concise summary
- Generate flashcards from content (keyword-based + optional Wikipedia lookup)
- Save summaries and flashcards into SQLite and review later

Dependencies
- Python 3.9+
- tkinter (built-in)
- requests
- PyPDF2

Install:
    pip install requests PyPDF2

Run:
    python study_helper.py
"""

import os
import re
import sqlite3
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import requests
from PyPDF2 import PdfReader

# --------------------------- Utility & NLP Helpers --------------------------- #

STOPWORDS = set(
    """
a an the and or but if while to of in on for with at by from up about into over after
then once here there all any both each few more most other some such no nor not only own same so than too very
can will just don should now is are was were be being been do does did having have has had this that these those
I you he she it we they them his her its our their as also between during against without within because until unless
""".split()
)

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str):
    text = clean_text(text)
    # simple sentence splitter
    parts = SENT_SPLIT_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


def tokenize(text: str):
    return [w.lower() for w in WORD_RE.findall(text)]


def frequency_summarize(text: str, max_sentences: int = 5) -> str:
    """Lightweight extractive summarizer using word frequency scoring."""
    sentences = split_sentences(text)
    if not sentences:
        return ""
    words = tokenize(text)
    if not words:
        return sentences[0][:300]

    freq = Counter(w for w in words if w not in STOPWORDS and len(w) > 2)
    if not freq:
        # fallback: first few sentences
        return " ".join(sentences[:max_sentences])

    sent_scores = []
    for idx, s in enumerate(sentences):
        score = sum(freq.get(w, 0) for w in tokenize(s)) / (len(tokenize(s)) + 1)
        # Mild positional bonus for earlier sentences
        score += 0.05 * max(0, (len(sentences) - idx) / len(sentences))
        sent_scores.append((score, idx, s))

    top = sorted(sent_scores, key=lambda x: x[0], reverse=True)[:max_sentences]
    # keep original order
    top_sorted = [s for _, _, s in sorted(top, key=lambda x: x[1])]
    return " ".join(top_sorted)


def extract_text_from_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        texts = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            texts.append(t)
        return clean_text("\n".join(texts))
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}")


def keyword_candidates(text: str, top_k: int = 12):
    words = [w for w in tokenize(text) if w not in STOPWORDS and len(w) > 3]
    common = Counter(words).most_common(top_k * 3)
    # filter very numeric-like or overly generic
    candidates = []
    for w, c in common:
        if not re.search(r"\d", w) and w not in {"data", "information", "example", "using"}:
            candidates.append((w, c))
        if len(candidates) >= top_k:
            break
    return [w for w, _ in candidates]


def find_sentence_for_term(text: str, term: str) -> str:
    term_l = term.lower()
    for s in split_sentences(text):
        if term_l in s.lower():
            return s
    return ""


def wikipedia_snippet(term: str, timeout: int = 8) -> str:
    """Return a short summary sentence for a term from Wikipedia (if available)."""
    try:
        # Use Wikipedia REST API summary endpoint
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(term)}"
        r = requests.get(url, timeout=timeout, headers={"accept": "application/json"})
        if r.status_code == 200:
            data = r.json()
            extract = data.get("extract") or ""
            if extract:
                # return first sentence
                first = split_sentences(extract)
                return first[0] if first else extract
    except Exception:
        pass
    return ""


def generate_flashcards(text: str, count: int = 10, use_wiki: bool = True):
    """Return list of (question, answer, term)."""
    text = clean_text(text)
    if not text:
        return []
    terms = keyword_candidates(text, top_k=count * 2)
    cards = []
    seen = set()
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        sent = find_sentence_for_term(text, term)
        ans = sent
        if use_wiki and (not ans or len(ans) < 25):
            wiki = wikipedia_snippet(term)
            if wiki:
                ans = wiki
        if not ans:
            # fallback short definition style
            ans = f"{term.capitalize()} is an important concept related to this topic."
        q = f"What is {term}?"
        cards.append((q, ans, term))
        if len(cards) >= count:
            break
    return cards

# --------------------------- Database Layer --------------------------- #

DB_PATH = "study_helper.db"

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS flashcards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deck_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    term TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(deck_id) REFERENCES decks(id) ON DELETE CASCADE
);
"""


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()


def save_summary(title: str, content: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO summaries (title, content, created_at) VALUES (?, ?, ?)",
        (title, content, datetime.utcnow().isoformat()),
    )
    conn.commit()
    sid = cur.lastrowid
    conn.close()
    return sid


def list_summaries():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, title, created_at FROM summaries ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


def load_summary(summary_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, title, content, created_at FROM summaries WHERE id=?", (summary_id,))
    row = cur.fetchone()
    conn.close()
    return row


def create_deck(name: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO decks (name, created_at) VALUES (?, ?)",
        (name, datetime.utcnow().isoformat()),
    )
    conn.commit()
    did = cur.lastrowid
    conn.close()
    return did


def list_decks():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, name, created_at FROM decks ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


def save_flashcards(deck_id: int, cards: list[tuple[str, str, str]]):
    conn = get_conn()
    cur = conn.cursor()
    for q, a, term in cards:
        cur.execute(
            "INSERT INTO flashcards (deck_id, question, answer, term, created_at) VALUES (?, ?, ?, ?, ?)",
            (deck_id, q, a, term, datetime.utcnow().isoformat()),
        )
    conn.commit()
    conn.close()


def load_deck_cards(deck_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, question, answer, term, created_at FROM flashcards WHERE deck_id=? ORDER BY id",
        (deck_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows

# --------------------------- Tkinter UI --------------------------- #

class StudyHelperApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Study Helper – Day 30 Project")
        self.geometry("1050x720")
        self.minsize(980, 640)
        self.configure(bg="#0b1020")

        init_db()

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background="#0b1020")
        style.configure("TLabel", background="#0b1020", foreground="#e7e9ee", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        style.configure("TNotebook", background="#0b1020")
        style.configure("TNotebook.Tab", padding=(12, 8))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.build_summarize_tab()
        self.build_flashcards_tab()
        self.build_saved_tab()

    # ---- Summarize Tab ---- #
    def build_summarize_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Summarize")

        top = ttk.Frame(tab)
        top.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(top, text="Title:").pack(side=tk.LEFT)
        self.sum_title_var = tk.StringVar(value="My Notes")
        ttk.Entry(top, textvariable=self.sum_title_var, width=40).pack(side=tk.LEFT, padx=6)

        ttk.Button(top, text="Load PDF", command=self.action_load_pdf).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Summarize", command=self.action_summarize).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Save Summary", command=self.action_save_summary).pack(side=tk.LEFT, padx=4)

        body = ttk.Frame(tab)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(left, text="Input Text / PDF Content").pack(anchor="w")
        self.input_text = tk.Text(left, wrap=tk.WORD, height=20, bg="#0f172a", fg="#e7e9ee", insertbackground="#e7e9ee")
        self.input_text.pack(fill=tk.BOTH, expand=True, pady=(4, 8))

        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        ttk.Label(right, text="Summary").pack(anchor="w")
        self.summary_text = tk.Text(right, wrap=tk.WORD, height=20, bg="#0f172a", fg="#e7e9ee", insertbackground="#e7e9ee")
        self.summary_text.pack(fill=tk.BOTH, expand=True, pady=(4, 8))

    def action_load_pdf(self):
        path = filedialog.askopenfilename(title="Choose PDF", filetypes=[["PDF", "*.pdf"]])
        if not path:
            return
        try:
            text = extract_text_from_pdf(path)
            if not text:
                messagebox.showwarning("No Text", "Couldn't extract text from this PDF. Try another.")
                return
            self.input_text.delete("1.0", tk.END)
            self.input_text.insert(tk.END, text)
            messagebox.showinfo("Loaded", f"Loaded text from: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def action_summarize(self):
        raw = self.input_text.get("1.0", tk.END).strip()
        if not raw:
            messagebox.showwarning("Empty", "Please paste text or load a PDF first.")
            return

        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, "Generating summary...")

        def worker():
            try:
                summary = frequency_summarize(raw, max_sentences=6)
            except Exception as e:
                summary = f"Failed to summarize: {e}"
            self.summary_text.after(0, lambda: self._set_summary(summary))

        threading.Thread(target=worker, daemon=True).start()

    def _set_summary(self, text):
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, text)

    def action_save_summary(self):
        title = self.sum_title_var.get().strip() or "Untitled"
        content = self.summary_text.get("1.0", tk.END).strip()
        if not content:
            messagebox.showwarning("Empty", "No summary to save.")
            return
        sid = save_summary(title, content)
        messagebox.showinfo("Saved", f"Saved summary #{sid} – '{title}'")
        self.refresh_saved_lists()

    # ---- Flashcards Tab ---- #
    def build_flashcards_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Flashcards")

        top = ttk.Frame(tab)
        top.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(top, text="Deck name:").pack(side=tk.LEFT)
        self.deck_name_var = tk.StringVar(value="My Deck")
        ttk.Entry(top, textvariable=self.deck_name_var, width=32).pack(side=tk.LEFT, padx=6)

        ttk.Label(top, text="Count:").pack(side=tk.LEFT, padx=(12, 0))
        self.card_count_var = tk.IntVar(value=10)
        ttk.Spinbox(top, from_=5, to=30, width=5, textvariable=self.card_count_var).pack(side=tk.LEFT)

        self.use_wiki_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Use Wikipedia help", variable=self.use_wiki_var).pack(side=tk.LEFT, padx=12)

        ttk.Button(top, text="Generate from Input", command=self.action_generate_cards).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Save Deck", command=self.action_save_deck).pack(side=tk.LEFT, padx=6)

        body = ttk.Frame(tab)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(left, text="Generated Flashcards (editable)").pack(anchor="w")
        self.cards_text = tk.Text(left, wrap=tk.WORD, bg="#0f172a", fg="#e7e9ee", insertbackground="#e7e9ee")
        self.cards_text.pack(fill=tk.BOTH, expand=True, pady=(4, 8))
        self.cards_text.insert(tk.END, "Each card on two lines:\nQ: ...\nA: ...\n---\n")

        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        ttk.Label(right, text="Preview / Tip").pack(anchor="w")
        self.cards_tip = tk.Text(right, height=8, wrap=tk.WORD, bg="#0f172a", fg="#9fb3ff", relief=tk.FLAT)
        self.cards_tip.pack(fill=tk.BOTH, expand=True, pady=(4, 8))
        self.cards_tip.insert(
            tk.END,
            "Tip: Edit questions/answers before saving. Use 'Use Wikipedia help' for crisper definitions.\n"
            "Format: Q: <question>\nA: <answer>\n--- (separator)\n",
        )
        self.cards_tip.configure(state=tk.DISABLED)

    def action_generate_cards(self):
        raw = self.input_text.get("1.0", tk.END).strip()
        if not raw:
            messagebox.showwarning("Empty", "Please paste text or load a PDF on the Summarize tab.")
            return

        count = max(5, min(30, int(self.card_count_var.get() or 10)))
        use_wiki = bool(self.use_wiki_var.get())
        self.cards_text.delete("1.0", tk.END)
        self.cards_text.insert(tk.END, "Generating flashcards...\n")

        def worker():
            try:
                cards = generate_flashcards(raw, count=count, use_wiki=use_wiki)
            except Exception as e:
                cards = [("Generation failed", str(e), "")] 
            def show():
                self.cards_text.delete("1.0", tk.END)
                buf = []
                for q, a, _ in cards:
                    buf.append(f"Q: {q}\nA: {a}\n---\n")
                self.cards_text.insert(tk.END, "".join(buf))
            self.cards_text.after(0, show)

        threading.Thread(target=worker, daemon=True).start()

    def parse_cards_text(self):
        text = self.cards_text.get("1.0", tk.END)
        parts = [p.strip() for p in text.split("---") if p.strip()]
        cards = []
        for p in parts:
            q_match = re.search(r"^\s*Q:\s*(.+)$", p, flags=re.IGNORECASE | re.MULTILINE)
            a_match = re.search(r"^\s*A:\s*(.+)$", p, flags=re.IGNORECASE | re.MULTILINE)
            if q_match and a_match:
                q = q_match.group(1).strip()
                a = a_match.group(1).strip()
                # derive a term from the Q (after 'What is ' if present)
                term = ""
                m = re.match(r"what\s+is\s+(.+?)\??$", q.lower())
                if m:
                    term = m.group(1).strip().title()
                cards.append((q, a, term))
        return cards

    def action_save_deck(self):
        name = self.deck_name_var.get().strip() or "My Deck"
        cards = self.parse_cards_text()
        if not cards:
            messagebox.showwarning("Empty", "No flashcards to save.")
            return
        deck_id = create_deck(name)
        save_flashcards(deck_id, cards)
        messagebox.showinfo("Saved", f"Saved deck #{deck_id} – '{name}' with {len(cards)} cards.")
        self.refresh_saved_lists()

    # ---- Saved Tab ---- #
    def build_saved_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Saved")

        main = ttk.Frame(tab)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Left column: summaries
        left = ttk.LabelFrame(main, text="Summaries")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.sum_list = tk.Listbox(left, height=18, bg="#0f172a", fg="#e7e9ee")
        self.sum_list.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.sum_list.bind("<<ListboxSelect>>", self.on_select_summary)

        self.sum_view = tk.Text(left, height=12, wrap=tk.WORD, bg="#0f172a", fg="#e7e9ee")
        self.sum_view.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 8))

        # Right column: decks
        right = ttk.LabelFrame(main, text="Flashcard Decks")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.deck_list = tk.Listbox(right, height=18, bg="#0f172a", fg="#e7e9ee")
        self.deck_list.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.deck_list.bind("<<ListboxSelect>>", self.on_select_deck)

        self.deck_view = tk.Text(right, height=12, wrap=tk.WORD, bg="#0f172a", fg="#e7e9ee")
        self.deck_view.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 8))

        self.refresh_saved_lists()

    def refresh_saved_lists(self):
        # Summaries
        self.sum_list.delete(0, tk.END)
        self._sum_rows = list_summaries()
        for sid, title, created in self._sum_rows:
            when = created.split("T")[0]
            self.sum_list.insert(tk.END, f"#{sid} · {title} · {when}")

        # Decks
        self.deck_list.delete(0, tk.END)
        self._deck_rows = list_decks()
        for did, name, created in self._deck_rows:
            when = created.split("T")[0]
            self.deck_list.insert(tk.END, f"#{did} · {name} · {when}")

        # Clear viewers
        self.sum_view.delete("1.0", tk.END)
        self.deck_view.delete("1.0", tk.END)

    def on_select_summary(self, event=None):
        sel = self.sum_list.curselection()
        if not sel:
            return
        idx = sel[0]
        sid = self._sum_rows[idx][0]
        row = load_summary(sid)
        if not row:
            return
        _, title, content, created = row
        self.sum_view.delete("1.0", tk.END)
        self.sum_view.insert(tk.END, f"Title: {title}\nCreated: {created}\n\n{content}")

    def on_select_deck(self, event=None):
        sel = self.deck_list.curselection()
        if not sel:
            return
        idx = sel[0]
        did = self._deck_rows[idx][0]
        cards = load_deck_cards(did)
        buf = []
        for _id, q, a, term, created in cards:
            buf.append(f"Q: {q}\nA: {a}\n---\n")
        self.deck_view.delete("1.0", tk.END)
        self.deck_view.insert(tk.END, "".join(buf))


if __name__ == "__main__":
    app = StudyHelperApp()
    app.mainloop()
