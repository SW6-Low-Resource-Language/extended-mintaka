# Purpose
This repository holds all the training/finetuning scripts that have been worked on / used in the process of evaluating the extensions of the mintaka dataset. 
Additionally the pipeline for evaluating is run through the evaluate_mintaka.py file, which produces data for hits@k and semantic similarity scores on inferenced LLM output. 

## Process
The process of evaluating mintaka consists of:
- Parsing LLM-Output into a structured format
- Process numerical and temporal answers to a standardized format that is comparable to the annotated format
- Calculate Hits@k and produce sheets with data
- Compute semantic similarity and produce sheets and visualizations
- 
## How To 

## env variables 
For the evaluation pipeline as is, to work you need to have the following key specificed:
- `GOOGLE_CLOUD_PROJECT`- ID for your project with the translator api enabled
  ~ This is used as part of the processing step of numerical answers, where certain answers are translated to english so we can run the text2num function on them to check for textual representations of numbers like "four", "two" etc.

## Running the evaluation script 
1. Specify the path to your inference data in the `generation_paths.json` file, we used the following configuration `"model_answers": {
        "path": "outputs/LLM_outputs/vllm_MODE_output_LANG.txt",
        "description": "Path to raw txt file containing model generation"
    }` here MODE and LANG indicate identifiers that can be used for dynamic replacement when running the script.
2. Extend the `comparative_dict.json` with approiate representations in the analysed language, the following mappings needs to be specified, e.g for danish: `"da": {
        "true_list": [
            "Sandt",
            "Rigtig",
            "Rigtigt",
            "Korrekt",
            "Ja"
        ],
        "false_list": [
            "Falsk",
            "Forkert",
            "Nej"
        ],
        "after": "Efter",
        "before": "Før",
        "same": "Samme",
        "less": "Mindre",
        "both": "Begge"`

3. Extend `questions_label_lang_dict.json` with the translated identifiers used for "question" and "answer"when running inference, in the evaluated language, currently it looks like this `{
    "question": {
        "da": "Spørgsmål",
        "bn": "প্রশ্ন",
        "en": "Question",
        "fi": "Kysymys"
    },
    "answer": {
        "da": "Svar",
        "bn": "উত্তর",
        "en": "Answer",
        "fi": "Vastaa"
    }
}

4. Set the `lang` and `mode` variables in `evaluate_mintaka.py`, default configuration is with `mode = zeroshot` (this can be changed when running inference after finetuning) and `lang` whatever language code you're evaluating on.. 


