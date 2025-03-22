# Text Generator

The repository contains an implementation of LSTM and Transformer models for generating text based on data from the book Pan Tadeusz. The models learn sequential dependencies in the text and are adapted to generate new fragments based on a given prompt.

---

## 🏗️ Model architecture

### LSTM-based Model (**LSTMTextGenerationModel**)

- A recurrent neural network (RNN) consisting of three LSTM layers:
  - **Embedding layer**: 64-dimensional representation of input characters
  - **LSTM layers**: three stacked layers with 256 hidden units each, using **dropout = 0.1**
  - **Fully connected output layer**: converts hidden state to character probabilities
- Optimized using **Adam optimizer**
- **CrossEntropyLoss** as the loss function
- **Early stopping** after 3 epochs without improvement

### Transformer-based Model (**TransformerTextGenerationModel**)

- Transformer decoder architecture:
  - **Embedding layer**: 128-dimensional character representation
  - **Positional encoding**: adds sequential dependencies to embeddings
  - **Decoder layers**: 2 layers with 8 attention heads each, hidden size 1024
  - **Dropout = 0.1**
  - **Fully connected output layer**: character-level prediction
- Optimized using **Adam optimizer**
- **CrossEntropyLoss** as the loss function
- **Early stopping** after 3 epochs without improvement

---

## 📊 Model results

### Results on test data:
#### LSTM model:
- **Loss**: 1.5911
- **Accuracy**: 50.58%
- **Top-5 Accuracy**: 82.92%

#### Transformer model:
- **Loss**: 1.9254
- **Accuracy**: 40.96%
- **Top-5 Accuracy**: 77.46%

### Example generated text:

#### LSTM-generated text:
```
Prompt: "litwo ojczyzno moja"
Generated: "litwo ojczyzno moja fale, francuza, choć ą pan sędzie siebie wyprawy..."
```

#### Transformer-generated text:
```
Prompt: "litwo ojczyzno moja"
Generated: "litwo ojczyzno moja m u  rś ewcrko ćd  ua ył  c  końce równy, gdybyś zamkał sypankę..."
```

Full generated text samples can be found in:
- **[LSTM samples](results/generated_samples/lstm_samples.txt)**
- **[Transformer samples](results/generated_samples/transformer_samples.txt)**

---

## 📂 Repository structure

text-generator\
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
│── results\
│   │── generated_samples\
│   │   │── lstm_samples.txt\
│   │   │ ── transformer_samples.txt\
│   │── evaluation_metrics.json\
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

- **LSTM training:**
  
   ```bash
   python training/train.py --model lstm
   ```
   
- **Transformer training:**

  ```bash
   python training/train.py --model transformer
   ```

  *Once training is complete, the model will be saved in saved_models/.*

### Text generation

   Once the model has been trained, it is possible to generate new text.

- **For LSTM:**

  ```bash
   python generation/generate_text.py --model lstm --prompt "W łaźniach publicznych bywał rzadko:" --num_chars 400
   ```

- **For Transformer:**

   ```bash
   python generation/generate_text.py --model transformer --prompt "W łaźniach publicznych bywał rzadko:" --num_chars 400
   ```

### Model evaluation

   During training, the model is automatically tested on the validation and test sets. The 'train.py' script will output **Test Loss**, **Accuracy**, and **Top-5 Accuracy**.

---

## ❗ Debugging

   If any errors occur:

1. Check if you have activated the Anaconda environment:

- On Windows:
     
   ```bash
   venv\Scripts\activate
   ```
   
- On Linux/macOS:
     
   ```bash
   source venv/bin/activate
   ```

2. Check if you have **PyTorch** with **CUDA**:

      ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
      
> **Note:** If it returns 'False', there may be a **problem with your CUDA installation**.
