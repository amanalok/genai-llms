# genai-llms
Repo for building and finetuning LLM applications

# Dialogue Summarization (NLP)
## Task Description
For the following dialogue within the prompt:
```
INPUT PROMPT:

Summarize the following conversation.

#Person1#: What do you want to know about me?
#Person2#: How about your academic records at college?
#Person1#: The average grade of all my courses is above 85.
#Person2#: In which subject did you get the highest marks?
#Person1#: In mathematics I got a 98.
#Person2#: Have you received any scholarships?
#Person1#: Yes, I have, and three times in total.
#Person2#: Have you been a class leader?
#Person1#: I have been a class commissary in charge of studies for two years.
#Person2#: Did you join in any club activities?
#Person1#: I was an aerobics team member in college.
#Person2#: What sport are you good at?
#Person1#: I am good at sprint and table tennis.
#Person2#: You are excellent.
```
The Baseline Human Summary is:
```
#Person2# asks #Person1# several questions, like academic records, the highest marks, scholarships, club activities, and skilled sports.
```
The below details about the dataset used for the task:
```
DatasetDict({
    train: Dataset({
        features: ['id', 'dialogue', 'summary', 'topic'],
        num_rows: 12460
    })
    validation: Dataset({
        features: ['id', 'dialogue', 'summary', 'topic'],
        num_rows: 500
    })
    test: Dataset({
        features: ['id', 'dialogue', 'summary', 'topic'],
        num_rows: 1500
    })
})
```
## Goal
Automate the task description with a language model. For this task, I have decided to use `FLAN-T5-Small` as the base model provided on the Huggingface Hub by Google.

## Zero Shot Ineference
For the above example prompt, the `FLAN-T5-Small` base model generated the below output with any tweaking to the model:
```
What is your school's average grade?
```

### Evaluation Results (Zero Shot)
The evaluation was done on 200 out of the 500 examples of the validation set due to memory constraints. Following are the `ROUGE` score results on the mentioned subset of validation set:
```
{'rouge1': 0.11584804033246703, 'rouge2': 0.03027225464065817, 'rougeL': 0.10352729474836227, 'rougeLsum': 0.10355547935185878}
```

## Fine-Tuning FLAN-T5-Small
### Full Finetuning
The model is full finetuned on the "Dialogue Summarization" dataset. Some high level details about the training are:
* epochs=5
* batch_size=2
* learning_rate=1e-5, weight_decay=0.01

Total Training Time: `1 hr, 30 mins, 54 seconds`

### Full Finetuned Model Inference
The inference on the prompt mentioned at the start of this document is below:
```
#Person1# tells #Person2# about #Person1#'s academic records at college. #Person1# got 98 marks in mathematics and has been a class commissary in charge of studies for two years. #Person1# has been a class commissary in charge of studies for two years. #Person1# is good at sprint and table tennis.
```

### Evaluation Results
Following are the `ROUGE` metric values on the subset of the validation set:
```
{'rouge1': 0.3749955899784231, 'rouge2': 0.1504537683072229, 'rougeL': 0.3038894433940812, 'rougeLsum': 0.30430939085755027}
```

### Parameter Efficient Fine Tuning (PEFT): LoRA
The training process uses PEFT based Low Rank Adaptation (LoRA) method for finetuning the base model `FLAN-T5-Small`. The advantage of using this methodology is that is trains much fewer number of parameters than full fine tuning.
So, while full finetuing the `FLAN-T5-Small` results in the following number of model parameters being re-trained:
```
Trainable model parameters:  76961152
All model parameters:  76961152
Percentage of trainable model parameters:  100.0
```
In case of using LoRA, following are the number of parameters trained:
```
Trainable model parameters:  344064
All model parameters:  77305216
Percentage of trainable model parameters:  0.445072166928555
```

Total Training Time: `1 hour 07 mins, 08 secs`
### LoRA Fine Tuned Model Inference
The inference on the prompt mentioned at the start of this document is below:



### Evaluation Results
Following are the `ROUGE` metric values on the subset of the validation set:
```
{'rouge1': 0.25434884287580434, 'rouge2': 0.08815030720527472, 'rougeL': 0.2212035268805954, 'rougeLsum': 0.22118046705264355}
```