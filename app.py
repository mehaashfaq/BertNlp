"""
Twitter Sentiment Analysis - BERT + PyQt5 GUI
Assignment 03 - NLP Lab

Loads a fine-tuned BERT model (saved via save_pretrained) and a Twitter
sentiment CSV dataset, lets the user click a tweet or type a sentence,
and displays the predicted sentiment with a confidence score.

Author: <your name>
"""

import sys
import os
import traceback

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QTextEdit, QFrame, QProgressBar, QMessageBox,
    QSplitter, QSizePolicy
)


# --------------------------------------------------------------------------
# Fallback label map, used only if the model's config does not already
# define id2label (Hugging Face models usually do after training if you
# set it before save_pretrained()).
# --------------------------------------------------------------------------
DEFAULT_LABEL_MAP = {0: "Irrelevant", 1: "Negative", 2: "Neutral", 3: "Positive"}


# --------------------------------------------------------------------------
# Worker threads (keep the GUI responsive during model loading / inference)
# --------------------------------------------------------------------------
class ModelLoaderThread(QThread):
    finished_ok = pyqtSignal(object, object, dict)
    failed = pyqtSignal(str)

    def __init__(self, model_dir):
        super().__init__()
        self.model_dir = model_dir

    def run(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            model.eval()

            id2label = getattr(model.config, "id2label", None)
            if id2label and len(id2label) > 0:
                # Normalize keys to int and make labels look nice
                label_map = {int(k): str(v).capitalize() for k, v in id2label.items()}
            else:
                label_map = DEFAULT_LABEL_MAP

            self.finished_ok.emit(model, tokenizer, label_map)
        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class PredictThread(QThread):
    finished_ok = pyqtSignal(str, float, list)
    failed = pyqtSignal(str)

    def __init__(self, model, tokenizer, label_map, text):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.text = text

    def run(self):
        try:
            inputs = self.tokenizer(
                self.text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1).squeeze(0)

            pred_id = int(torch.argmax(probs).item())
            confidence = float(probs[pred_id].item()) * 100
            label = self.label_map.get(pred_id, str(pred_id))
            all_probs = [
                (self.label_map.get(i, str(i)), float(p.item()) * 100)
                for i, p in enumerate(probs)
            ]
            self.finished_ok.emit(label, confidence, all_probs)
        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


# --------------------------------------------------------------------------
# Main Window
# --------------------------------------------------------------------------
class SentimentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.label_map = DEFAULT_LABEL_MAP
        self.df = None

        self.setWindowTitle("Twitter Sentiment Analysis  |  BERT + PyQt")
        self.resize(1280, 780)
        self.setMinimumSize(1100, 680)

        self._build_ui()
        self._apply_styles()

    # ---------------------------------------------------------------- UI --
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        # ---------- Header ----------
        header = QFrame()
        header.setObjectName("headerCard")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(20, 16, 20, 16)

        title = QLabel("Twitter Sentiment Analysis")
        title.setObjectName("titleLabel")
        subtitle = QLabel("BERT-powered sentiment classification  •  load a dataset, load your trained model, click a tweet")
        subtitle.setObjectName("subtitleLabel")

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        root.addWidget(header)

        # ---------- Toolbar (buttons + status) ----------
        toolbar = QFrame()
        toolbar.setObjectName("card")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(16, 12, 16, 12)
        toolbar_layout.setSpacing(12)

        self.btn_load_dataset = QPushButton("📂  Load Dataset CSV")
        self.btn_load_model = QPushButton("🤖  Load BERT Model")
        self.btn_predict_selected = QPushButton("🔮  Predict Selected")

        for b in (self.btn_load_dataset, self.btn_load_model, self.btn_predict_selected):
            b.setCursor(Qt.PointingHandCursor)
            b.setMinimumHeight(40)

        self.btn_load_dataset.clicked.connect(self.load_dataset)
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_predict_selected.clicked.connect(self.predict_selected_row)
        self.btn_predict_selected.setEnabled(False)

        self.status_label = QLabel("⬤ No model  •  ⬤ No dataset")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        toolbar_layout.addWidget(self.btn_load_dataset)
        toolbar_layout.addWidget(self.btn_load_model)
        toolbar_layout.addWidget(self.btn_predict_selected)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.status_label)

        root.addWidget(toolbar)

        # ---------- Main split: table | prediction panel ----------
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Left: dataset table
        left_card = QFrame()
        left_card.setObjectName("card")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(16, 16, 16, 16)

        left_title = QLabel("Loaded Tweets")
        left_title.setObjectName("cardTitle")
        left_layout.addWidget(left_title)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["#", "Tweet Text", "Actual Label"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.cellClicked.connect(self.on_row_clicked)
        left_layout.addWidget(self.table)

        # Right: prediction panel
        right_card = QFrame()
        right_card.setObjectName("card")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(10)

        right_title = QLabel("Prediction Panel")
        right_title.setObjectName("cardTitle")
        right_layout.addWidget(right_title)

        sel_label = QLabel("Selected Sentence")
        sel_label.setObjectName("fieldLabel")
        self.selected_text = QTextEdit()
        self.selected_text.setReadOnly(True)
        self.selected_text.setFixedHeight(80)
        self.selected_text.setPlaceholderText("Click a tweet from the table, or type below…")

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setObjectName("divider")

        pred_title = QLabel("Predicted Sentiment")
        pred_title.setObjectName("fieldLabel")
        self.prediction_box = QLabel("—")
        self.prediction_box.setObjectName("predictionBox")
        self.prediction_box.setAlignment(Qt.AlignCenter)
        self.prediction_box.setMinimumHeight(56)

        conf_title = QLabel("Confidence")
        conf_title.setObjectName("fieldLabel")
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setFormat("%p%")
        self.confidence_bar.setTextVisible(True)

        self.breakdown_label = QLabel("")
        self.breakdown_label.setObjectName("breakdownLabel")
        self.breakdown_label.setWordWrap(True)

        right_layout.addWidget(sel_label)
        right_layout.addWidget(self.selected_text)
        right_layout.addWidget(divider)
        right_layout.addWidget(pred_title)
        right_layout.addWidget(self.prediction_box)
        right_layout.addWidget(conf_title)
        right_layout.addWidget(self.confidence_bar)
        right_layout.addWidget(self.breakdown_label)
        right_layout.addStretch()

        # Manual input area
        manual_title = QLabel("Type a Sentence")
        manual_title.setObjectName("fieldLabel")
        self.manual_input = QTextEdit()
        self.manual_input.setFixedHeight(70)
        self.manual_input.setPlaceholderText("Type any sentence to analyze its sentiment…")

        self.btn_predict_manual = QPushButton("✨  Predict Typed Sentence")
        self.btn_predict_manual.setCursor(Qt.PointingHandCursor)
        self.btn_predict_manual.setMinimumHeight(40)
        self.btn_predict_manual.clicked.connect(self.predict_manual_text)
        self.btn_predict_manual.setEnabled(False)

        right_layout.addWidget(manual_title)
        right_layout.addWidget(self.manual_input)
        right_layout.addWidget(self.btn_predict_manual)

        splitter.addWidget(left_card)
        splitter.addWidget(right_card)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        root.addWidget(splitter, stretch=1)

        # ---------- Footer status bar ----------
        self.statusBar().showMessage("Ready. Load a model and dataset to begin.")

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f1420;
            }
            QLabel, QWidget {
                color: #e7ebf3;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            #headerCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1c2740, stop:1 #24365c);
                border-radius: 14px;
            }
            #titleLabel {
                font-size: 24px;
                font-weight: 700;
                color: #ffffff;
            }
            #subtitleLabel {
                font-size: 12.5px;
                color: #a9b6d1;
                margin-top: 2px;
            }
            #card {
                background-color: #161d2e;
                border-radius: 14px;
                border: 1px solid #232d45;
            }
            #cardTitle {
                font-size: 15px;
                font-weight: 600;
                color: #ffffff;
                margin-bottom: 6px;
            }
            #fieldLabel {
                font-size: 11.5px;
                font-weight: 600;
                color: #8ea0c7;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            #statusLabel {
                font-size: 12.5px;
                color: #a9b6d1;
            }
            #divider {
                background-color: #232d45;
                max-height: 1px;
                border: none;
            }
            #predictionBox {
                background-color: #1e2a44;
                border: 1px solid #2e3c5e;
                border-radius: 10px;
                font-size: 20px;
                font-weight: 700;
                color: #ffffff;
            }
            #breakdownLabel {
                font-size: 11.5px;
                color: #8ea0c7;
            }
            QPushButton {
                background-color: #2f6fed;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #4a83f2;
            }
            QPushButton:pressed {
                background-color: #2559c4;
            }
            QPushButton:disabled {
                background-color: #2a334a;
                color: #6d7996;
            }
            QTableWidget {
                background-color: #10182a;
                gridline-color: #232d45;
                border: 1px solid #232d45;
                border-radius: 8px;
                selection-background-color: #2f6fed;
                selection-color: #ffffff;
            }
            QHeaderView::section {
                background-color: #1c2740;
                color: #cdd7ec;
                padding: 6px;
                border: none;
                font-weight: 600;
            }
            QTextEdit {
                background-color: #10182a;
                border: 1px solid #232d45;
                border-radius: 8px;
                padding: 6px;
                color: #e7ebf3;
            }
            QProgressBar {
                background-color: #10182a;
                border: 1px solid #232d45;
                border-radius: 8px;
                text-align: center;
                color: #ffffff;
                height: 22px;
            }
            QProgressBar::chunk {
                background-color: #2f6fed;
                border-radius: 8px;
            }
            QStatusBar {
                background-color: #0f1420;
                color: #8ea0c7;
            }
        """)

    # ------------------------------------------------------------ actions --
    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Twitter Sentiment CSV", "", "CSV Files (*.csv)"
        )
        if not path:
            return

        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read CSV:\n{e}")
            return

        text_col = self._find_column(df, ["text", "tweet", "tweet_text", "sentence"])
        label_col = self._find_column(df, ["sentiment", "label", "target", "class"])

        if text_col is None:
            QMessageBox.critical(
                self, "Missing Column",
                "Could not find a text/tweet column in this CSV.\n"
                "Expected one of: text, tweet, tweet_text, sentence."
            )
            return

        self.df = df
        self.text_col = text_col
        self.label_col = label_col

        self.table.setRowCount(0)
        for i, row in df.iterrows():
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(str(i + 1)))
            self.table.setItem(r, 1, QTableWidgetItem(str(row[text_col])))
            actual = str(row[label_col]) if label_col else "—"
            self.table.setItem(r, 2, QTableWidgetItem(actual))

        self._update_status()
        self.statusBar().showMessage(f"Loaded {len(df)} rows from {os.path.basename(path)}")

    @staticmethod
    def _find_column(df, candidates):
        cols_lower = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand in cols_lower:
                return cols_lower[cand]
        return None

    def load_model(self):
        model_dir = QFileDialog.getExistingDirectory(self, "Select Saved BERT Model Folder")
        if not model_dir:
            return

        self.btn_load_model.setEnabled(False)
        self.statusBar().showMessage("Loading model, please wait…")

        self.loader_thread = ModelLoaderThread(model_dir)
        self.loader_thread.finished_ok.connect(self._on_model_loaded)
        self.loader_thread.failed.connect(self._on_model_failed)
        self.loader_thread.start()

    def _on_model_loaded(self, model, tokenizer, label_map):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.btn_load_model.setEnabled(True)
        self.btn_predict_manual.setEnabled(True)
        self.btn_predict_selected.setEnabled(self.df is not None)
        self._update_status()
        self.statusBar().showMessage("Model loaded successfully.")

    def _on_model_failed(self, error_msg):
        self.btn_load_model.setEnabled(True)
        QMessageBox.critical(self, "Model Load Failed", error_msg)
        self.statusBar().showMessage("Model load failed.")

    def _update_status(self):
        model_ok = self.model is not None
        data_ok = self.df is not None
        model_dot = "🟢" if model_ok else "🔴"
        data_dot = "🟢" if data_ok else "🔴"
        self.status_label.setText(
            f"{model_dot} Model {'loaded' if model_ok else 'not loaded'}   "
            f"{data_dot} Dataset {'loaded' if data_ok else 'not loaded'}"
        )

    def on_row_clicked(self, row, _col):
        text_item = self.table.item(row, 1)
        if text_item:
            self.selected_text.setPlainText(text_item.text())

    def predict_selected_row(self):
        text = self.selected_text.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "No Sentence Selected", "Click a tweet in the table first.")
            return
        self._run_prediction(text)

    def predict_manual_text(self):
        text = self.manual_input.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Empty Input", "Type a sentence first.")
            return
        self.selected_text.setPlainText(text)
        self._run_prediction(text)

    def _run_prediction(self, text):
        if self.model is None or self.tokenizer is None:
            QMessageBox.warning(self, "No Model Loaded", "Please load a trained BERT model first.")
            return

        self.btn_predict_selected.setEnabled(False)
        self.btn_predict_manual.setEnabled(False)
        self.statusBar().showMessage("Predicting…")

        self.predict_thread = PredictThread(self.model, self.tokenizer, self.label_map, text)
        self.predict_thread.finished_ok.connect(self._on_prediction_done)
        self.predict_thread.failed.connect(self._on_prediction_failed)
        self.predict_thread.start()

    def _on_prediction_done(self, label, confidence, all_probs):
        self.btn_predict_selected.setEnabled(True)
        self.btn_predict_manual.setEnabled(True)

        colors = {
            "Positive": "#1fbf6b",
            "Negative": "#e5484d",
            "Neutral": "#c9a227",
            "Irrelevant": "#7a8bb8",
        }
        color = colors.get(label, "#2f6fed")
        self.prediction_box.setText(label.upper())
        self.prediction_box.setStyleSheet(f"""
            #predictionBox {{
                background-color: {color}22;
                border: 1px solid {color};
                border-radius: 10px;
                font-size: 20px;
                font-weight: 700;
                color: {color};
            }}
        """)
        self.confidence_bar.setValue(int(round(confidence)))

        breakdown = "  |  ".join(f"{lbl}: {p:.1f}%" for lbl, p in all_probs)
        self.breakdown_label.setText(breakdown)

        self.statusBar().showMessage(f"Prediction: {label} ({confidence:.1f}% confidence)")

    def _on_prediction_failed(self, error_msg):
        self.btn_predict_selected.setEnabled(True)
        self.btn_predict_manual.setEnabled(True)
        QMessageBox.critical(self, "Prediction Failed", error_msg)
        self.statusBar().showMessage("Prediction failed.")


# --------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = SentimentApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
