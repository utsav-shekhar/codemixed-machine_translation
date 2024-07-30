# Code-Mixed Machine Translation Project

This project aims to advance the field of machine translation by focusing on translating fromEnglish to Hinglish (code mixed variant of language used by bilingual speakers of Hindi and English). Our approach involves experimenting with various architectures and strategies, including GRU and LSTM with attention mechanisms, sequence-to-sequence transformers, and few-shot learning on large language models.

Drive Link : https://iiitaphyd-my.sharepoint.com/:f:/g/personal/sankalp_bahad_research_iiit_ac_in/EhGrdI6wjgRMqtJvgZkRuu0BAkVvv1zjvXLu8St2u9ihPA?e=LBDIlZ

Presentation Link : https://docs.google.com/presentation/d/1e4THowfDgD55vh69wYXXAWXz5XeXqNixkYfyx0VmYTg/edit?usp=sharing

## Team Members

- Utsav Shekhar
- Sankalp Bahad
- Yash Bhaskar

## Course

- Advanced Natural Language Processing (ANLP)

## Institute

- IIIT Hyderabad

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Model Inference](#model-inference)
- [Performance Metrics](#performance-metrics)
- [Contribution](#contribution)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

To set up the project environment to run the Jupyter Notebooks, follow these steps:

bash
git clone https://github.com/TheTriad/CodeMixed-Translation.git
cd CodeMixed-Translation
pip install -r requirements.txt


## Dataset

The dataset statistics are detailed in our project report and the dataset can be accessed [here](https://ritual.uh.edu/lince/datasets). Preprocessing is done using the CSNLI library for transliteration and text normalization.

## Model Training

Each model has a separate Jupyter Notebook for the training process. Navigate to the `training_notebooks` directory to find all the training notebooks.

### Instructions to Run Training Notebooks

1. Ensure all dependencies from `requirements.txt` are installed.
2. Download the dataset and place it in the `data` folder.
3. Open the Jupyter Notebook for the model you wish to train.
4. Run each cell in the notebook to start the training process.

## Model Inference

After training the models, use the inference notebooks located in the `inference_notebooks` directory to perform translations.

### Instructions for Inference

1. Load the trained model weights.
2. Run the inference notebook corresponding to the model.
3. Input the Hinglish sentence when prompted to get the English translation.

## Performance Metrics

We use Sacre BLEU scores for evaluating the translation quality. The metrics for each model are detailed in the project report.

## Contribution

Guidelines for contributing to this project:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- [CSNLI Library](https://aclanthology.org/2021.calcs-1.7.pdf)
