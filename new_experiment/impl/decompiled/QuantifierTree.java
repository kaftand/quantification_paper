/*
 * Decompiled with CFR 0.151.
 * 
 * Could not load the following classes:
 *  j48quantify.BinC45Split
 *  j48quantify.C45Split
 *  j48quantify.Distribution
 *  j48quantify.InfoGainDistanceSplitCrit
 *  j48quantify.ModelSelectionQuantify
 *  j48quantify.QuantifierSplitModel
 *  weka.core.Capabilities
 *  weka.core.CapabilitiesHandler
 *  weka.core.Drawable
 *  weka.core.Instance
 *  weka.core.Instances
 *  weka.core.RevisionHandler
 *  weka.core.RevisionUtils
 *  weka.core.Utils
 */
package j48quantify;

import j48quantify.BinC45Split;
import j48quantify.C45Split;
import j48quantify.Distribution;
import j48quantify.InfoGainDistanceSplitCrit;
import j48quantify.ModelSelectionQuantify;
import j48quantify.QuantifierSplitModel;
import java.io.Serializable;
import java.util.LinkedList;
import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;

public class QuantifierTree
implements Drawable,
Serializable,
CapabilitiesHandler,
RevisionHandler {
    static final long serialVersionUID = -8722249377542734193L;
    protected ModelSelectionQuantify m_toSelectModel;
    protected QuantifierSplitModel m_localModel;
    protected QuantifierTree[] m_sons;
    protected boolean m_isLeaf;
    protected boolean m_isEmpty;
    protected Instances m_train;
    protected Distribution m_test;
    protected int m_id;
    protected double m_distanceManhattan;
    protected int m_idxClass;
    protected double[] m_vectorError;
    protected double[] m_errorSubtree;
    private static long PRINTED_NODES = 0L;

    protected static long nextID() {
        return PRINTED_NODES++;
    }

    protected static void resetID() {
        PRINTED_NODES = 0L;
    }

    public QuantifierTree(ModelSelectionQuantify toSelectLocModel) {
        this.m_toSelectModel = toSelectLocModel;
    }

    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities((CapabilitiesHandler)this);
        result.enableAll();
        return result;
    }

    public void buildClassifier(Instances data) throws Exception {
        this.getCapabilities().testWithFail(data);
        data = new Instances(data);
        data.deleteWithMissingClass();
        this.buildTree(data, false);
    }

    public void buildTree(Instances data, boolean keepData) throws Exception {
        double tmpDistance = 0.0;
        int numClasses = data.numClasses();
        if (keepData) {
            this.m_train = data;
        }
        this.m_test = null;
        this.m_isLeaf = false;
        this.m_isEmpty = false;
        this.m_sons = null;
        this.m_localModel = this.m_toSelectModel.selectModel(data, this.m_vectorError, this.m_idxClass, this.m_distanceManhattan);
        if (this.m_localModel.numSubsets() > 1) {
            int attIndex = this.m_localModel instanceof C45Split ? ((C45Split)this.m_localModel).attIndex() : ((BinC45Split)this.m_localModel).attIndex();
            this.m_vectorError = InfoGainDistanceSplitCrit.getError((Instances)data, (Distribution)this.m_localModel.distribution(), (double[])this.m_vectorError, (int)this.m_idxClass, (int)attIndex);
            this.m_distanceManhattan = 0.0;
            int i = 0;
            while (i < this.m_vectorError.length) {
                this.m_distanceManhattan += Math.pow(this.m_vectorError[i], 2.0);
                ++i;
            }
            Instances[] localInstances = this.m_localModel.split(data);
            data = null;
            this.m_sons = new QuantifierTree[this.m_localModel.numSubsets()];
            i = 0;
            while (i < this.m_sons.length) {
                int k;
                this.m_idxClass = this.m_localModel.m_distribution.maxClass(i);
                this.m_sons[i] = this.getNewTree(localInstances[i]);
                if (this.m_sons[i].m_localModel.m_numSubsets != 1) {
                    this.m_vectorError = this.m_sons[i].m_vectorError;
                    k = 0;
                    while (k < this.m_vectorError.length) {
                        tmpDistance += Math.pow(this.m_vectorError[k], 2.0);
                        ++k;
                    }
                    this.m_distanceManhattan = tmpDistance;
                    tmpDistance = 0.0;
                }
                localInstances[i] = null;
                k = 0;
                while (k < numClasses) {
                    this.m_errorSubtree[k] = this.m_errorSubtree[k] + this.m_sons[i].m_errorSubtree[k];
                    ++k;
                }
                ++i;
            }
        } else {
            this.m_isLeaf = true;
            this.m_errorSubtree[this.m_localModel.m_distribution.maxClass()] = (int)this.m_localModel.m_distribution.total();
            if (Utils.eq((double)data.sumOfWeights(), (double)0.0)) {
                this.m_isEmpty = true;
            }
            data = null;
        }
    }

    public void buildTree(Instances train, Instances test, boolean keepData) throws Exception {
        int numClasses = train.numClasses();
        if (keepData) {
            this.m_train = train;
        }
        this.m_isLeaf = false;
        this.m_isEmpty = false;
        this.m_sons = null;
        this.m_localModel = this.m_toSelectModel.selectModel(train, test, this.m_vectorError, this.m_idxClass, this.m_distanceManhattan);
        this.m_test = new Distribution(test, this.m_localModel);
        if (this.m_localModel.numSubsets() > 1) {
            Instances[] localTrain = this.m_localModel.split(train);
            Instances[] localTest = this.m_localModel.split(test);
            test = null;
            train = null;
            int attIndex = this.m_localModel instanceof C45Split ? ((C45Split)this.m_localModel).attIndex() : ((BinC45Split)this.m_localModel).attIndex();
            this.m_vectorError = InfoGainDistanceSplitCrit.getError((Instances)train, (Distribution)this.m_localModel.distribution(), (double[])this.m_vectorError, (int)this.m_idxClass, (int)attIndex);
            this.m_distanceManhattan = 0.0;
            int k = 0;
            while (k < this.m_vectorError.length) {
                this.m_distanceManhattan += Math.pow(this.m_vectorError[k], 2.0);
                ++k;
            }
            this.m_sons = new QuantifierTree[this.m_localModel.numSubsets()];
            int i = 0;
            while (i < this.m_sons.length) {
                this.m_idxClass = this.m_localModel.m_distribution.maxClass(i);
                this.m_sons[i] = this.getNewTree(localTrain[i], localTest[i]);
                if (this.m_sons[i].m_localModel.m_numSubsets != 1) {
                    attIndex = this.m_sons[i].m_localModel instanceof C45Split ? ((C45Split)this.m_sons[i].m_localModel).attIndex() : ((BinC45Split)this.m_sons[i].m_localModel).attIndex();
                    this.m_vectorError = InfoGainDistanceSplitCrit.getError((Instances)localTrain[i], (Distribution)this.m_sons[i].m_localModel.m_distribution, (double[])this.m_vectorError, (int)this.m_idxClass, (int)attIndex);
                    this.m_distanceManhattan = 0.0;
                    k = 0;
                    while (k < this.m_vectorError.length) {
                        this.m_distanceManhattan += Math.pow(this.m_vectorError[k], 2.0);
                        ++k;
                    }
                }
                localTrain[i] = null;
                localTest[i] = null;
                k = 0;
                while (k < numClasses) {
                    this.m_errorSubtree[k] = this.m_errorSubtree[k] + this.m_sons[i].m_errorSubtree[k];
                    ++k;
                }
                ++i;
            }
        } else {
            this.m_isLeaf = true;
            this.m_errorSubtree[this.m_localModel.m_distribution.maxClass()] = (int)this.m_localModel.m_distribution.total();
            if (Utils.eq((double)train.sumOfWeights(), (double)0.0)) {
                this.m_isEmpty = true;
            }
            test = null;
            train = null;
        }
    }

    public double classifyInstance(Instance instance) throws Exception {
        double maxProb = -1.0;
        int maxIndex = 0;
        int j = 0;
        while (j < instance.numClasses()) {
            double currentProb = this.getProbs(j, instance, 1.0);
            if (Utils.gr((double)currentProb, (double)maxProb)) {
                maxIndex = j;
                maxProb = currentProb;
            }
            ++j;
        }
        return maxIndex;
    }

    public final void cleanup(Instances justHeaderInfo) {
        this.m_train = justHeaderInfo;
        this.m_test = null;
        if (!this.m_isLeaf) {
            int i = 0;
            while (i < this.m_sons.length) {
                this.m_sons[i].cleanup(justHeaderInfo);
                ++i;
            }
        }
    }

    public final double[] distributionForInstance(Instance instance, boolean useLaplace) throws Exception {
        double[] doubles = new double[instance.numClasses()];
        int i = 0;
        while (i < doubles.length) {
            doubles[i] = !useLaplace ? this.getProbs(i, instance, 1.0) : this.getProbsLaplace(i, instance, 1.0);
            ++i;
        }
        return doubles;
    }

    public int assignIDs(int lastID) {
        int currLastID;
        this.m_id = currLastID = lastID + 1;
        if (this.m_sons != null) {
            int i = 0;
            while (i < this.m_sons.length) {
                currLastID = this.m_sons[i].assignIDs(currLastID);
                ++i;
            }
        }
        return currLastID;
    }

    public int graphType() {
        return 1;
    }

    public String graph() throws Exception {
        StringBuffer text = new StringBuffer();
        this.assignIDs(-1);
        text.append("digraph J48Tree {\n");
        if (this.m_isLeaf) {
            text.append("N" + this.m_id + " [label=\"" + this.m_localModel.dumpLabel(0, this.m_train) + "\" " + "shape=box style=filled ");
            if (this.m_train != null && this.m_train.numInstances() > 0) {
                text.append("data =\n" + this.m_train + "\n");
                text.append(",\n");
            }
            text.append("]\n");
        } else {
            text.append("N" + this.m_id + " [label=\"" + this.m_localModel.leftSide(this.m_train) + "\" ");
            if (this.m_train != null && this.m_train.numInstances() > 0) {
                text.append("data =\n" + this.m_train + "\n");
                text.append(",\n");
            }
            text.append("]\n");
            this.graphTree(text);
        }
        return String.valueOf(text.toString()) + "}\n";
    }

    public String prefix() throws Exception {
        StringBuffer text = new StringBuffer();
        if (this.m_isLeaf) {
            text.append("[" + this.m_localModel.dumpLabel(0, this.m_train) + "]");
        } else {
            this.prefixTree(text);
        }
        return text.toString();
    }

    public StringBuffer[] toSource(String className) throws Exception {
        StringBuffer[] result = new StringBuffer[2];
        if (this.m_isLeaf) {
            result[0] = new StringBuffer("    p = " + this.m_localModel.distribution().maxClass(0) + ";\n");
            result[1] = new StringBuffer("");
        } else {
            StringBuffer text = new StringBuffer();
            StringBuffer atEnd = new StringBuffer();
            long printID = QuantifierTree.nextID();
            text.append("  static double N").append(String.valueOf(Integer.toHexString(this.m_localModel.hashCode())) + printID).append("(Object []i) {\n").append("    double p = Double.NaN;\n");
            text.append("    if (").append(this.m_localModel.sourceExpression(-1, this.m_train)).append(") {\n");
            text.append("      p = ").append(this.m_localModel.distribution().maxClass(0)).append(";\n");
            text.append("    } ");
            int i = 0;
            while (i < this.m_sons.length) {
                text.append("else if (" + this.m_localModel.sourceExpression(i, this.m_train) + ") {\n");
                if (this.m_sons[i].m_isLeaf) {
                    text.append("      p = " + this.m_localModel.distribution().maxClass(i) + ";\n");
                } else {
                    StringBuffer[] sub = this.m_sons[i].toSource(className);
                    text.append(sub[0]);
                    atEnd.append(sub[1]);
                }
                text.append("    } ");
                if (i == this.m_sons.length - 1) {
                    text.append('\n');
                }
                ++i;
            }
            text.append("    return p;\n  }\n");
            result[0] = new StringBuffer("    p = " + className + ".N");
            result[0].append(String.valueOf(Integer.toHexString(this.m_localModel.hashCode())) + printID).append("(i);\n");
            result[1] = text.append(atEnd);
        }
        return result;
    }

    public int numLeaves() {
        int num = 0;
        if (this.m_isLeaf) {
            return 1;
        }
        int i = 0;
        while (i < this.m_sons.length) {
            num += this.m_sons[i].numLeaves();
            ++i;
        }
        return num;
    }

    public int numNodes() {
        int no = 1;
        if (!this.m_isLeaf) {
            int i = 0;
            while (i < this.m_sons.length) {
                no += this.m_sons[i].numNodes();
                ++i;
            }
        }
        return no;
    }

    public String toString() {
        try {
            StringBuffer text = new StringBuffer();
            if (this.m_isLeaf) {
                text.append(": ");
                text.append(this.m_localModel.dumpLabel(0, this.m_train));
            } else {
                this.dumpTree(0, text);
            }
            text.append("\n\nNumber of Leaves  : \t" + this.numLeaves() + "\n");
            text.append("\nSize of the tree : \t" + this.numNodes() + "\n");
            return text.toString();
        }
        catch (Exception e) {
            return "Can't print classification tree.";
        }
    }

    protected QuantifierTree getNewTree(Instances data) throws Exception {
        QuantifierTree newTree = new QuantifierTree(this.m_toSelectModel);
        newTree.m_distanceManhattan = this.m_distanceManhattan;
        newTree.m_idxClass = this.m_idxClass;
        newTree.m_vectorError = this.m_vectorError;
        newTree.m_errorSubtree = new double[data.numClasses()];
        newTree.buildTree(data, false);
        return newTree;
    }

    protected QuantifierTree getNewTree(Instances train, Instances test) throws Exception {
        QuantifierTree newTree = this.getNewTree(train);
        newTree.buildTree(train, test, false);
        return newTree;
    }

    private void dumpTree(int depth, StringBuffer text) throws Exception {
        int i = 0;
        while (i < this.m_sons.length) {
            text.append("\n");
            int j = 0;
            while (j < depth) {
                text.append("|   ");
                ++j;
            }
            text.append(this.m_localModel.leftSide(this.m_train));
            text.append(this.m_localModel.rightSide(i, this.m_train));
            if (this.m_sons[i].m_isLeaf) {
                text.append(": ");
                text.append(this.m_localModel.dumpLabel(i, this.m_train));
            } else {
                this.m_sons[i].dumpTree(depth + 1, text);
            }
            ++i;
        }
    }
