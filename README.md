# Text Generator

Repozytorium zawiera implementację modeli LSTM i Transformer do generowania tekstu na podstawie danych z książki Pan Tadeusz. Modele uczą się sekwencyjnych zależności w tekście i są przystosowane do generowania nowych fragmentów na podstawie podanego promptu.

---

## 🏗️ Model architecture

---

## 📊 Model results

---

## 📂 Repository structure

text-generaotr\
│── models\
│  │── lstm_model.py\
│  │── transformer_model.py\
│ \
│── training\
│  │── train.py\
│  │── utils.py\
│ \
│── generation\
│  │── generate_text.py\
│ \
│── data\
│  │── pantadeusz.txt\
│ \
│── saved_models\
│  │── best_lstm.pth\
│  │── best_transformer.pth\
│ \
│── README.md\
│── .gitignore\
│── requirements.txt\

---

## 🚀 Installation

1. **Clone repository:**

   ```bash
   git clone https://github.com/t-piwowarski/text-generator.git
   cd text-generator

2. **Create and activate a virtual environment (optional but recommended):**
   
- On Windows:
     
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   
- On Linux/macOS:
     
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages:**
   
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Running models

### Training

  To train the model, use the script 'train.py'. You can choose which model should be trained, **LSTM** or **Transformer**.

- **LSTM Training:**
  
   ```bash
   python training/train.py --model lstm
   ```
   
- **Transformer Training:**

  ```bash
   python training/train.py --model transformer
   ```

  *Once training is complete, the model will be saved in saved_models/.*
