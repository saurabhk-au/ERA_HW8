# ERA_HW6_CNN
CNN 
     # CIFAR 10 dataset

   This project trains a convolutional neural network on the CIFAR-10 dataset using PyTorch and Albumentations.
   
  ## Model Details
     Use  
     - **Depthwise Separable Convolutions:** Yes
     - **Adaptive Avg Pool:** Yes   
     - **Total Parameter Count: <=  200k
     - **Batch Normalization:** Yes
     - **DropOut:** Yes
     - **Fully Connected Layer or GAP:** Yes

     ## Tests

     - **Total Parameter Count Test:** Pass (less than 200,000)
     - **Test Accuracy:** Pass (greater than 85%%)


## Project Structure

- `models/cifar10_model.py`: Contains the model architecture.
- `utils/augmentations.py`: Contains data augmentation functions.
- `train_and_test.py`: Script to train and test the model.

## Results

The model aims to achieve an accuracy of 85% on the CIFAR-10 test set.

## License

This project is licensed under the MIT License.
