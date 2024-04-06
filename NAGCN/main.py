import load_data
from model import GCN
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from scipy import interp
import time
import visdom
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc

def calculate_performace(num, y_pred, y_prob, y_test):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(num):
        if y_test[index] ==1:
            if y_test[index] == y_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if y_test[index] == y_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    acc = float(tp + tn)/num
    try:
        precision = float(tp)/(tp + fp)
        recall = float(tp)/ (tp + fn)
        f1_score = float((2*precision*recall)/(precision+recall))
        MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
        sens = tp/(tp+fn)
    except ZeroDivisionError:
        print("You can't divide by 0.")
        precision=recall=f1_score =sens = MCC=100
    AUC = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test,y_prob)
    return tp, fp, tn, fn, acc, precision, sens, f1_score, MCC, AUC,auprc

def NAGCN():
    dataset, cd_pairs = load_data.dataset()
    kf = KFold(n_splits=5, shuffle=True)
    model = GCN()
    ave_acc = 0
    ave_prec = 0
    ave_sens = 0
    ave_f1_score = 0
    ave_mcc = 0
    ave_auc = 0
    ave_auprc = 0
    localtime = time.asctime(time.localtime(time.time()))
    with open('results/result.txt', 'a') as f:
        f.write('time:\t' + str(localtime) + "\n")
        for train_index, test_index in kf.split(cd_pairs):
            c_dmatix, train_cd_pairs, test_cd_pairs = load_data.C_Dmatix(cd_pairs, train_index, test_index)
            dataset['c_d'] = c_dmatix
            score, cir_fea, dis_fea = load_data.feature_representation(model,dataset)
            train_dataset = load_data.new_dataset(cir_fea, dis_fea, train_cd_pairs)
            test_dataset = load_data.new_dataset(cir_fea, dis_fea, test_cd_pairs)

            X_train, y_train = train_dataset[:, :-2], train_dataset[:, -2:]
            X_test, y_test = test_dataset[:, :-2], test_dataset[:, -2:]
            clf = RandomForestClassifier(n_estimators=200, n_jobs=11, max_depth=20)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_pred = y_pred[:, 0]
            y_prob = clf.predict_proba(X_test)
            y_prob = y_prob[1][:, 0]
            tp, fp, tn, fn, acc, prec, sens, f1_score, MCC, AUC, AUPRC= calculate_performace(len(y_pred), y_pred, y_prob, y_test[:, 0])

            print('RF: \n  Acc = \t', acc, '\n  prec = \t', prec, '\n  sens = \t', sens, '\n  f1_score = \t', f1_score,
                  '\n  MCC = \t', MCC, '\n  AUC = \t', AUC, '\n  AUPRC = \t', AUPRC)
            f.write('RF: \t  tp = \t' + str(tp) + '\t fp = \t' + str(fp) + '\t tn = \t' + str(tn) + '\t fn = \t' + str(
                fn) + '\t  Acc = \t' + str(acc) + '\t  prec = \t' + str(prec) + '\t  sens = \t' + str(
                sens) + '\t  f1_score = \t' + str(f1_score) + '\t  MCC = \t' + str(MCC) + '\t  AUC = \t' + str(
                AUC) + '\t  AUPRC = \t' + str(AUPRC) + '\n')
            ave_acc += acc
            ave_prec += prec
            ave_sens += sens
            ave_f1_score += f1_score
            ave_mcc += MCC
            ave_auc += AUC
            ave_auprc += AUPRC

        ave_acc /= n_fold
        ave_prec /= n_fold
        ave_sens /= n_fold
        ave_f1_score /= n_fold
        ave_mcc /= n_fold
        ave_auc /= n_fold
        ave_auprc /= n_fold
        print('Final: \t  tp = \t' + str(tp) + '\t fp = \t' + str(fp) + '\t tn = \t' + str(tn) + '\t fn = \t' + str(
            fn) + '\t  Acc = \t' + str(ave_acc) + '\t  prec = \t' + str(ave_prec) + '\t  sens = \t' + str(
            ave_sens) + '\t  f1_score = \t' + str(ave_f1_score) + '\t  MCC = \t' + str(ave_mcc) + '\t  AUC = \t' + str(
            ave_auc) + '\t  AUPRC = \t' + str(ave_auprc) + '\n')
        f.write('Final: \t  tp = \t' + str(tp) + '\t fp = \t' + str(fp) + '\t tn = \t' + str(tn) + '\t fn = \t' + str(
            fn) + '\t  Acc = \t' + str(ave_acc) + '\t  prec = \t' + str(ave_prec) + '\t  sens = \t' + str(
            ave_sens) + '\t  f1_score = \t' + str(ave_f1_score) + '\t  MCC = \t' + str(ave_mcc) + '\t  AUC = \t' + str(
            ave_auc) + '\t  AUPRC = \t' + str(ave_auprc) + '\n')

if __name__ == "__main__":
    NAGCN()