""" 
This file is created for the group-assignment for the course
Advanced Programming & Biomedical Data Analysis (8CC00).
This utils code file contains the objects and functions
supporting the software training and applying the 
Machine learning models selected.

"""
# calculation/data structuring/data conversions 
import numpy as np
import pandas as pd
from tkinter import filedialog as fd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw, Descriptors, AllChem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
IPythonConsole.ipython_useSVG=True

# ML
import numpy as np
from sklearn import neighbors, metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef
import joblib


# import data

def importFiles(nrFiles=int, writetxt=True):
    """ Import .csv files by selection
    Parameter(s):   nrFiles: (int) number of files to be selected
    Returns:        allAHDL1InhibitorsDf (pd.DataFrame) dataframe containing all SMILES and inhibit info
        """
    allAHDL1Inhibitors = []
    for nr, _ in enumerate(range(nrFiles)):
        if nr == 0:
            filename = fd.askopenfilename()
            AHDL1Inhibitors = pd.read_csv(filename ,header = None)
            allAHDL1Inhibitors.append(AHDL1Inhibitors)
        else: 
            filename = fd.askopenfilename()
            AHDL1Inhibitors = pd.read_csv(filename ,header = None)
            # drop header of the consecutive files
            AHDL1Inhibitors = AHDL1Inhibitors.iloc[1:, :]
            # create list of separate pd.dataframes
            allAHDL1Inhibitors.append(AHDL1Inhibitors)
    allAHDL1InhibitorsDf = pd.concat(allAHDL1Inhibitors)
    allAHDL1InhibitorsDf = allAHDL1InhibitorsDf.reset_index(drop=True)
    return allAHDL1InhibitorsDf

def writeNewMolfile(AHDL1InhibitorDf, filename='AllTestedMols.txt'):
    """ write SMILES in 'AHDL1InhibitorDf' dataframe to a .txt file
    Parameter(s):       AHDL1InhibitorDf: dataframe containing SMILES and inhibit property
                        fileName: (str) name for the file to be written
    Returns:            -
    """
    allTestedMolecules = AHDL1InhibitorDf[0] # first 3 for testing, needs to change for all molecules (remove[0:4])
    MolList = allTestedMolecules.values.tolist()
    with open(f'{filename}', 'w') as fp:
        for mol in MolList:
            # write each item on a new line
            fp.write("%s\n" % mol)
        print(f'All molecules stored in: {filename}')

    
def getMolDescriptors(mol, missingVal=None):
    """ Calculate the full list of descriptors for a molecule
    Parameter(s):   mol:  current molecule
                    missingValue: (=None) is set to given value if descriptor cannot be calculated
    Returns:        res: (dict): a dictionary containing all descriptor data 
    """
    res = {}
    for nm,fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res

def createDescriptorDf(filename='AllTestedMols.txt'):
    """
    """
    RDLogger.DisableLog('rdApp.*')
    suppl = Chem.SmilesMolSupplier('AllTestedMols.txt')
    mols = [m for m in suppl]
    allDescrs = [getMolDescriptors(m) for m in mols]
    return pd.DataFrame(allDescrs)


def generateMorganFingerprint(filename='AllTestedMols.txt'):
    """
    Parameter(s):   filename: (str) file containing SMILES of different molecules

    Return:         x: (numpy.array) array containing fingerprint test/train dataset
    """
    RDLogger.DisableLog('rdApp.*')
    suppl = Chem.SmilesMolSupplier(filename)
    mols = [m for m in suppl]
    fingerp = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in mols]

    fingerprint = []
    for f in fingerp:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(f, arr)
        fingerprint.append(arr) 
        x = np.asarray(fingerprint)
    return x 

def convertToMol(filename='AllTestedMols.txt'):
    """ 
    Parameter(s):

    Return:
    """
    suppl = Chem.SmilesMolSupplier(filename)
    mols = [m for m in suppl]
    RDLogger.DisableLog('rdApp.*')
    allDescrs = [getMolDescriptors(m) for m in mols]
    return pd.DataFrame(allDescrs)

def CombineDescriptorsAndFigerprints(Morganfingerprints, descriptors):
    """
    """
    x = np.concatenate((Morganfingerprints, descriptors), axis=1)
    return x, x.shape

# preprocessing

def scaleData(data, scaletype='standardize'):
    """
    """
    if scaletype == 'standardize':
        scale = StandardScaler().fit(data)
        scaledData = scale.transform(data)    
    elif scaletype == 'normalize':
        scale = MinMaxScaler().fit(data)
        scaledData = scale.transform(data)  
    else:
        raise ValueError("input should be 'standardize' or 'Normalize'")
    return scaledData

def PCAfeatureReduction(data, varianceThreshold):
    """
    """
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    else: 
        df = data

    # df_std = StandardScaler().fit_transform(df)
    # df_std =  pd.DataFrame(df_std)

    pca = PCA()
    principalComponents = pca.fit_transform(df)
    principalDF = pd.DataFrame(data=principalComponents)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    i = 0
    sumVar = 0
    while sumVar<varianceThreshold:
        sumVar = cumulative_variance_ratio[i]
        i +=1

    return df.iloc[:,:i], principalDF


def getTargetData(AHDL1Inhibitors):
    """
    
    """
    y = AHDL1Inhibitors[1][1:].astype(int)
    imbalanceIndex = sum(y)/len(y)
    return y, imbalanceIndex

def splitTrainTestData(x,y, test_size=0.20, nSplits=5, seed=13, printSplit=False):
    """
    
    """
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=test_size, random_state=seed)
    # create folds for cross- validation
    cv = StratifiedKFold(n_splits=nSplits, shuffle=True,random_state=seed)
    if printSplit == True:
        for i, (train_index, test_index) in enumerate(cv.split(xTrain, yTrain)):
            print("\nFold_" + str(i+1))
            print("TRAIN:", train_index)
            print("TEST:", test_index)
    return xTrain, xTest, yTrain, yTest, cv
 
 # RF

def trainRF(xTrain, yTrain, cv):
    """
    """
    
    RFc = RandomForestClassifier()


    param_grid = {"max_features": [xTrain.shape[1] // 10, xTrain.shape[1] // 7, 
                               xTrain.shape[1] // 5, xTrain.shape[1] // 3], 
                  "n_estimators": [100, 250, 500]}
    
    scorer = metrics.make_scorer(metrics.balanced_accuracy_score)
    RFgrid = GridSearchCV(RFc, param_grid, scoring=scorer, error_score="raise",
                           return_train_score=True, n_jobs=2, cv=cv, verbose=1)
    
    # train model
    trainedRFC = RFgrid.fit(xTrain, yTrain)
    print('Best Random Forest Classifier: \n', 'Parameters:', RFgrid.best_params_,
           '\n balanced accuracy: ', RFgrid.best_score_)
    return trainedRFC, RFgrid.best_params_, RFgrid, param_grid, RFgrid.best_score_


# SVM 

def trainSVC(xTrain, yTrain, cv):
    """
    """
    svc = SVC(kernel='rbf', probability=True)
    
    param_grid = {"C": [10 ** i for i in range(0, 5)],
                    "gamma": [10 ** i for i in range(-6, 0)]}
    
    scorer = metrics.make_scorer(metrics.balanced_accuracy_score)
    SVCgrid = GridSearchCV(svc, param_grid, scoring=scorer, error_score="raise",
                           return_train_score=True, n_jobs=2, cv=cv, verbose=1)
    
    # train model
    trainedSVC = SVCgrid.fit(xTrain, yTrain)
    print('Best Random Forest Classifier: \n', 'Parameters:', SVCgrid.best_params_,
           '\n balanced accuracy: ', SVCgrid.best_score_)
    return trainedSVC, SVCgrid.best_params_, SVCgrid, param_grid, SVCgrid.best_score_


# KNN
def trainKnn(xTrain, yTrain, cv, k_range=list):
    """
    """
    knn = KNeighborsClassifier()

    param_grid = {'n_neighbors': k_range,
                    'weights': ('uniform', 'distance'),
                    }
    scorer = metrics.make_scorer(metrics.balanced_accuracy_score)
    # defining parameter range
    
    knngrid = GridSearchCV(knn, param_grid, scoring=scorer, error_score="raise",
                           return_train_score=True, n_jobs=2, cv=cv, verbose=1)

    trainedknn = knngrid.fit(xTrain, yTrain)
    print('Best Random Forest Classifier: \n', 'Parameters:', knngrid.best_params_,
           '\n balanced accuracy: ', knngrid.best_score_)
    return trainedknn, knngrid.best_params_, knngrid, param_grid, knngrid.best_score_


# saving and testing
def saveTrainedModel(model, filename):
    """
    """
    joblib.dump(model, f"{filename}.pkl", compress=3)

def testTrainedModel(xTest, yTest, model=None, savedModelfilename=str, scaledData=False, scaledDatafile=None):
    """
    """
    if model == None:
        if savedModelfilename.find('.pkl') != -1:
            trainedModel = joblib.load(f"{savedModelfilename}")
        else:
            trainedModel = joblib.load(f"{savedModelfilename}.pkl")
    else:
        trainedModel = model

    if scaledData == True:
        pred = trainedModel.predict(xTest) 
    elif scaledDatafile != None:
        scale = joblib.load(scaledDatafile)
        # scale descriptors of the test set compounds
        xTest = scale.transform(xTest)
        # predict logBB class
        pred = trainedModel.predict(xTest)

    balAcc = metrics.balanced_accuracy_score(yTest, pred)
    pred_prob = trainedModel.predict_proba(xTest)
    return pred, balAcc, pred_prob

def thresholdedAccuracy(yTest, pred, pred_prob, threshold=0.8):
    """
    """
    # calc maximum predicted probability for each row (compound) and compare to the threshold
    da = np.amax(pred_prob, axis=1) > threshold
    threshAcc = accuracy_score(np.asarray(yTest)[da], pred[da])
    # calc coverage
    coverage = sum(da) / len(da)
    return da, threshAcc, coverage
