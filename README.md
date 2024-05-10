# Model Comparison: LeNet-5 vs AlexNet vs VGG-16 on CIFAR-10

This repository contains code for comparing the performance of three popular convolutional neural network (CNN) architectures: LeNet-5, AlexNet, and VGG-16, on the CIFAR-10 dataset. The CIFAR-10 dataset is a common benchmark for image classification tasks that consists of 60,000 32x32 color images in 10 classes.

## Purpose

The purpose of this project is to evaluate and compare the performance of different CNN architectures on the CIFAR-10 dataset. By analyzing metrics such as accuracy, precision, recall, and F1-score, I aim to gain insights into the strengths and weaknesses of each architecture in classifying images from the CIFAR-10 dataset.

## Project Structure

- **models/**: Directory containing Python files for each CNN architecture implementation.
  - **lenet5.py**: Implementation of LeNet-5.
  - **alexnet.py**: Implementation of AlexNet.
  - **vgg16.py**: Implementation of VGG-16.
- **notebooks/**: Directory containing Jupyter notebooks for model comparison and evaluation.
  - **Model_Comparison.ipynb**: Notebook for comparing the performance of LeNet-5, AlexNet, and VGG-16.
- **results/**: Directory for storing result files such as classification reports and confusion matrices.
- **requirements.txt**: File listing dependencies required to run the code.
- **LICENSE**: License file.

## Getting Started 

To run the code in this repository, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies listed in `requirements.txt`. You can install them using the following command:

    ```bash
    pip install -r requirements.txt
    ```

3. Open the Jupyter notebook `Model_Comparison.ipynb` in the `notebooks/` directory and execute each code cell sequentially by pressing Shift + Enter. Follow the instructions and comments provided in each code cell for guidance.

4. Analyze the results in the notebook to compare the performance of LeNet-5, AlexNet, and VGG-16 on the CIFAR-10 dataset. Pay attention to metrics such as accuracy, precision, recall, and F1-score.

5. Feel free to explore the notebook and draw your own conclusions. If you have any suggestions for future work or improvements, I welcome your input.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- The CIFAR-10 dataset is provided by the Canadian Institute for Advanced Research (CIFAR).
- Implementation inspiration and guidance from various online resources and tutorials.
