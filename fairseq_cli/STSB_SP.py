import os
import argparse
from fairseq.models.roberta import RobertaModel
from scipy.stats import pearsonr, spearmanr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="STS-B Pearson/Spearman calculator")
    parser.add_argument("--data_root", type=str, default='./glue')
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    roberta = RobertaModel.from_pretrained(
        f'{args.model_path}',
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=f'{args.data_root}/data-bin/STS-B-bin'
    )

    roberta.cuda()
    roberta.eval()
    gold, pred = [], []
    with open(f'{args.data_root}/STS-B/dev.tsv', encoding='utf-8') as fin:
        fin.readline()
        for index, line in enumerate(fin):
            tokens = line.strip().split('\t')
            sent1, sent2, target = tokens[7], tokens[8], float(tokens[9])
            tokens = roberta.encode(sent1, sent2)
            features = roberta.extract_features(tokens)
            predictions = 5.0 * roberta.model.classification_heads['sentence_classification_head'](features)
            gold.append(target)
            pred.append(predictions.item())

    print('| Pearson: ', pearsonr(gold, pred))
    print('| Spearman: ', spearmanr(gold, pred))