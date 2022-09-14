import os
import argparse
from fairseq.models.roberta import RobertaModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MRPC F1 calculator")
    parser.add_argument("--data_root", type=str, default='./glue')
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    roberta = RobertaModel.from_pretrained(
        f'{args.model_path}',
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=f'{args.data_root}/data-bin/MRPC-bin'
    )

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )
    ncorrect, nsamples = 0, 0
    predictions = []
    ground_truth = []
    roberta.cuda()
    roberta.eval()
    with open(f'{args.data_root}/MRPC/dev.tsv') as fin:
        fin.readline()
        for index, line in enumerate(fin):
            tokens = line.strip().split('\t')
            sent1, sent2, target = tokens[3], tokens[4], tokens[0]
            tokens = roberta.encode(sent1, sent2)
            prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
            prediction_label = label_fn(prediction)
            ncorrect += int(prediction_label == target)
            prediction_label = int(prediction_label)
            target = int(target)
            predictions.append(prediction_label)
            ground_truth.append(target)
            nsamples += 1
    tp, fp, tn, fn = 0, 0, 0, 0
    for idx, (pred, tgt) in enumerate(zip(predictions, ground_truth)):
        if pred and tgt:
            tp += 1
        elif pred and not tgt:
            fp += 1
        elif not pred and tgt:
            fn += 1
        else:
            tn += 1
    F1 = tp / (tp + 1/2*(fp+fn))
    Acc = (tp+tn) / (tp+tn+fp+fn)
    assert abs(Acc - float(ncorrect)/float(nsamples)) < 0.01, "acc calculation is wrong"
    print('| Accuracy: ', float(ncorrect)/float(nsamples))
    print('| F1: ', F1)