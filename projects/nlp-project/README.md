# Natural Language Processing Project

For this project, you will be required to apply all previous knowledge of NLP with deep learning. All projects must include a README.md file and all code must include docstrings according to NumPy standard. Additionally the Neural Network must be in PyTorch. The project must include the following:

1. An original per group dataset of text either to generate new text based on the dataset or to perform a classification task. Provide an explanation of the dataset (**Due: 2019-09-18**):
   * Where does it come from?
   * What is your use case?
   * Why did you choose this dataset?
   * Cite any previous use cases for the dataset
2. Process the dataset to a representation that a Neural Network can take as input. You have complete freedom on how to do this; you don’t necessarily have to do it the same way we’ve been doing it in class, as long as the output can be properly used by the Neural Network. Provide an explanation of the step by step procedure to clean the data. (**Due: 2019-09-23**)
3. Perform the task you chose (generation, classification) using a loss function and architecture that is appropriate for your task. Explain: (**Due: 2019-09-25**)
   * What is the task?
   * Why is the loss function you chose the correct one?
   * The neural network architecture
4. Explain the model results. 
   * How did the loss behave during training?
   * For classification tasks: How accurate is your model?
   * For text generation: How good is the text generated? Does it make sense? Can you tell it’s from the dataset you used? How so?
   * The model should achieve reasonable performance. Difficulty of the task will be taken into account when evaluating this.
5. Code must be reproducible. We should be able to run it and obtain the same results as you did. Use random seeds for this.


## Bonus Points:
* Use a dataset in spanish
* Apply software engineering best practices:
  * Clean, readable, modular code separated into different modules according to use case.
  * Program works with a CLI.
  * Include tests for your code.
* Host and serve model predictions.
