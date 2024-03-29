# Language Agnostic Syllabification with Active Learning

This repository contains an implementation of a language-agnostic syllabification method using active learning. Syllabification is the process of splitting a word into syllables, crucial in speech synthesis and recognition. Our approach utilizes active learning to reduce the need for large labeled datasets. By adapting the neural network from [Krantz et al. (2019)](https://arxiv.org/abs/1909.13362) and training it with active learning, we improved accuracy on the Portuguese and Italian datasets, using only a small fraction of the data: 384 words (1.4% of the dataset) for Portuguese and 528 words (0.6% of the dataset) for Italian.

## 🚀 Usage

### Prerequisites
Before running the project, ensure that you have the following:
- MATLAB 2021a (or a newer version)
- Statistics and Machine Learning Toolbox
- Text Analytics Toolbox

### Running the Project
- Clone this repository to your local machine or download the ZIP archive.
- Open MATLAB and navigate to the root directory of the cloned repository.
- Locate the src folder and open the `main.m` file.
- Run the `main.m` script to execute the project.

## 📊 Results
The project showcases its effectiveness by achieving remarkable accuracy values with minimal labeled data. Specifically, the following results were obtained:

- *Porlex v3* (Portuguese dataset): Achieved an accuracy of **96.8%** using only **384 words**, which corresponds to **1.4%** of the original dataset.
- *PhonItalia* (Italian dataset): Achieved an accuracy of **82.0%** using only **528 words**, which corresponds to **0.6%** of the original dataset.
- *Lexique 2* (French dataset): Achieved an accuracy of **95.8%** using only **208 words**, which is **less than 0.01%** of the whole dataset.

For both Portuguese and Italian, these results surpass those achieved by training the network on the entire dataset, 95.6% and 81%, respectively.

## 📜 License
This project is licensed under the [MIT License](LICENSE).

## 🎉 Acknowledgments
We would like to acknowledge the work of [Krantz et al. (2019)](https://arxiv.org/abs/1909.13362) for providing the neural network architecture used in this project. Their research serves as a foundation for our active learning adaptation.

## 📬 Contact
If you have any questions, suggestions, or just want to say hello, feel free to email me at [jmcabralpinto@gmail.com](mailto:jmcabralpinto@gmail.com).
