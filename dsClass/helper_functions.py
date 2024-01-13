import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn import tree
import itertools
import copy
import sys
from sklearn.externals.six import StringIO
import pydotplus


def write_Tree(fname_outdir, clf, cols_features):
    # Write rules to file
    # REMEMBER: Rules refer ONLY to the TRAINING data!
    fname_DT_rules = fname_outdir + r'\DT_rules_output.txt'
    
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    get_code(clf, cols_features) # get_code() is a function from helper_functions.py
    sys.stdout = old_stdout
    # end capture
    to_text = mystdout.getvalue()
    # write file
    text_file = open(fname_DT_rules, "w")
    text_file.write(to_text)
    text_file.close()

    # Using pyplotplus and graphviz (both must be installed on the computer for this bit to work) 
    # in order to visualize the decision tree

    # ERROR!! there's some bug with deep trees (max_depth > 5)
    outfile= fname_outdir + r'\tree.dot'
    pngfile= fname_outdir + r'\tree.png'
    dot_data = StringIO()  
    tree.export_graphviz(clf, out_file=outfile,  
                         feature_names=cols_features,  
                         class_names=['Bad', 'Good'],  
                         filled=True, rounded=True,  
                         special_characters=True)  

    graph = pydotplus.graph_from_dot_file(outfile)
    graph.write_png(pngfile)

def calculate_loss(pred, label, cfn, cfp, ctp=0, ctn=0):
    # test = class probability
    # truth = true label
    # c** = the loss associated with fn, fp, tp, tn
    #       in our simplified example we'll use only cfn and cfp, while the others are set to 0 by default
	#
	# returns: a dataframe with TN, FN, FP, TP, loss, per prediction-threshold
    
    predicted_prob = copy.deepcopy(pred)
	
    i = np.arange(len(label)) 
    col_label = 'label'
    col_pred = 'prediction'
    loss_matrices = pd.DataFrame({col_label : np.array(label), col_pred : pd.Series(pred)})
    loss_matrices.sort_values([col_pred], ascending=0, inplace=True)
    loss_matrices.reset_index(drop=True,inplace=True)    

    loss_matrices['TN'] = 0
    loss_matrices['FP'] = 0
    loss_matrices['FN'] = 0
    loss_matrices['TP'] = 0
    loss_matrices['loss'] = 0

    thresholds = list(loss_matrices[col_pred])
    pop = len(label)

    for i, thr in enumerate(thresholds):
        
        predicted_prob[pred >  thr] = 1
        predicted_prob[pred <= thr] = 0
        conf_mat = metrics.confusion_matrix(label, predicted_prob)
        conf_mat = list(conf_mat.flatten())

        tn = conf_mat[0]
        fp = conf_mat[1]
        fn = conf_mat[2]
        tp = conf_mat[3]
		
        loss_matrices.loc[i, 'TN']= tn
        loss_matrices.loc[i, 'FP']= fp
        loss_matrices.loc[i, 'FN']= fn
        loss_matrices.loc[i, 'TP']= tp
        #loss_matrices.loc[i, 'loss'] = (cfn*fn + cfp*fp + ctp*tp + ctn*tn)/pop
        loss_matrices.loc[i, 'loss'] = (cfn*fn + cfp*fp + ctp*tp + ctn*tn)
    loss_matrices.__delitem__(col_label)
    return(loss_matrices)

def plot_confusion_matrix(cm, label_names, title='Confusion matrix', cmaptype=0):
    # Pretty plotting for confusion matrices
    if (cmaptype == 0):
        cmap=plt.cm.Purples
    elif (cmaptype == 1):
        cmap=plt.cm.Greens
    else:
        cmap=plt.cm.Reds
        
    fontsz = 12
    plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=fontsz+4)
    plt.colorbar()
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, fontsize=fontsz+2)
    ax = plt.gca()
    ax.set_xticklabels((ax.get_xticks() ).astype(str))
    ax.grid(False)
    plt.yticks(tick_marks, fontsize=fontsz+2)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], weight = 'bold',
                 horizontalalignment="center", fontsize=fontsz+2,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=fontsz+2)
    plt.xlabel('Predicted label', fontsize=fontsz+2)

def cfm_convention():
    # just to remember the convensions of confusion matrix. It IS confusing!! :)
    print("\t\tPredicted")
    print("\t\t0\t1")
    print("Actual\t0\tTN\tFP")
    print("\t1\tFN\tTP")

def plot_ROC(fpr, tpr, fontsz, title):

	df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
	plt.figure(figsize=[10,7])
	auc = metrics.auc(fpr,tpr) 
	plt.plot(fpr, tpr, linewidth=4, label='ROC curve (area = %0.2f)' % auc)

	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate', fontsize=fontsz+2)
	plt.ylabel('True Positive Rate', fontsize=fontsz+2)
	plt.xticks(fontsize=fontsz)
	plt.yticks(fontsize=fontsz)
	plt.title(title, fontsize=fontsz+4)
	plt.legend(loc="lower right", fontsize=fontsz+2)
	plt.show()    

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()
		
def get_code(tree, feature_names, tabdepth=0):
# prints decision tree rules with proper tabbing
        left      = tree.tree_.children_left
        right     = tree.tree_.children_right
        threshold = tree.tree_.threshold
        features  = [feature_names[i] for i in tree.tree_.feature]
        value = tree.tree_.value

        def recurse(left, right, threshold, features, node, tabdepth=0):
                if (threshold[node] != -2):
                        print ('\t' * tabdepth, end="")
                        print ("if ( " + features[node] + " <= " + str(threshold[node]) + " ) {")
                        if left[node] != -1:
                                recurse (left, right, threshold, features,left[node], tabdepth+1)
                        print ('\t' * tabdepth, end="")
                        print ("} else {")
                        if right[node] != -1:
                                recurse (left, right, threshold, features,right[node], tabdepth+1)
                        print ('\t' * tabdepth,end="")
                        print ("}")
                else:
                        print ('\t' * tabdepth,end="")
                        print ("return " + str((value[node])))

        recurse(left, right, threshold, features, 0)