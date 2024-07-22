import torch


class Accuracy():
    
    """
    Average classification accuracy.
    """

    def __init__(self, classes) -> None:
        
        self.num_classes = classes
        self.reset()
        
        
    def __str__(self) -> str:
        
        """
        Return a string representation of the performance, accuracy and per class accuracy.
        """

        overall_accuracy = self.accuracy()
        per_class_accuracy = self.per_class_accuracy()
        
        # Create a representation string
        performance_str = f"Overall Accuracy: {overall_accuracy:.4f}\n"
        
        # Calculate the mean per class accuracy
        performance_str += f"Per-Class Accuracy: {per_class_accuracy:.4f}\n"
        
        # Get the maximum class name length
        max_class_name_length = max([len(class_label) for class_label in ['Axe', 'Book', 'Hammer']])
        
        for idx, class_label in enumerate(['Axe', 'Book', 'Hammer']):
            
            class_accuracy = self.correct_per_class[idx] / self.total_per_class[idx]
            performance_str += f"   - Accuracy Class {class_label:<{max_class_name_length}}: {class_accuracy:.4f}\n"
        
        return performance_str


    def reset(self) -> None:

        """
        Resets the internal state.
        """

        self.total_predictions = 0
        self.correct_predictions = 0
        
        # Initialize a list to track correct predictions per class
        self.correct_per_class = [0] * self.num_classes
        
        # Initialize a list to track total predictions per class
        self.total_per_class = [0] * self.num_classes


    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:

        """
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        """

        # Get the predicted class for each instance
        predicted_classes = torch.argmax(prediction, dim=1)

        # Check predictions and targets
        if prediction.shape[0] != target.shape[0]:
            raise ValueError('Data shape is unsupported')
        
        if prediction.shape[1] != self.num_classes:
            raise ValueError('Data shape is unsupported')
        
        if not torch.all(target >= 0) or not torch.all(target < self.num_classes):
            raise ValueError('Data values are unsupported')
        
        # Update the counters
        for i in range(target.shape[0]):
            
            self.total_predictions += 1
            self.total_per_class[target[i]] += 1
            
            # Check if the prediction matches the target
            if predicted_classes[i] == target[i]:
                self.correct_predictions += 1
                self.correct_per_class[target[i]] += 1


    def accuracy(self) -> float:
        
        """
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """

        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions
            
    
    def per_class_accuracy(self) -> float:
        
        """
        Compute and return the per class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """
        
        per_class_accs = [
                            (self.correct_per_class[i] / self.total_per_class[i]) if self.total_per_class[i] > 0 else 0.0
                            for i in range(self.num_classes)
        ]
        
        if sum(per_class_accs) == 0:
            return 0.0
        else:
            return sum(per_class_accs) / self.num_classes