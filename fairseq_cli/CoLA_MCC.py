import os
import argparse
from sklearn.metrics import matthews_corrcoef
from fairseq.models.roberta import RobertaModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CoLA MCC calculator")
    parser.add_argument("--data_root", type=str, default='./glue')
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    roberta = RobertaModel.from_pretrained(
        f'{args.model_path}',
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=f'{args.data_root}/data-bin/CoLA-bin'
    )

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )
    ncorrect, nsamples = 0, 0
    predictions = []
    ground_truth = []
    roberta.cuda()
    roberta.eval()
    with open(f'{args.data_root}/CoLA/dev.tsv', encoding='utf-8') as fin:
        fin.readline()
        for index, line in enumerate(fin):
            tokens = line.strip().split('\t')
            sent, target = tokens[3], tokens[1]
            tokens = roberta.encode(sent)
            prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
            prediction_label = label_fn(prediction)
            ncorrect += int(prediction_label == target)
            prediction_label = int(prediction_label)
            target = int(target)
            predictions.append(prediction_label)
            ground_truth.append(target)
            nsamples += 1

    print('| Accuracy: ', float(ncorrect)/float(nsamples))
    MCC = matthews_corrcoef(ground_truth, predictions)
    print('| MCC: ', MCC)