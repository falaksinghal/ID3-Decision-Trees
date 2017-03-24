/* 
 * Machine Learning Assignment -1
 * Implementation of decision Tree algorithm Using VI and Gain Heuristics
 * Implementation of post pruning heuristics
 * by - Falak Singhal
 * NetID - fxs161530@utdallas.edu
 * University of Texas at Dallas
 * 
 * */ 

package classes;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

enum NodeType {
	BRANCH, LEAF 	// ENUM for labeling node as a Branch or  a Leaf

};

class Node{
	int value;
	double positivecount = 0;
	double negativecount = 0;
	String attributeName;
	NodeType type;
	double entropyfinal;
	Node leftchild = null;
	Node rightchild = null;

	public Node(NodeType Type, Double Entropy, String attr, int value) {
		this.type = Type;
		this.entropyfinal = Entropy;
		Node leftchild = null;
		Node rightchild = null;
		this.attributeName = attr;
		this.value = value;
	}

	public Node(NodeType Type, Double Entropy, String attr, int value, Node leftchild, Node rightchild,
			double positivecount, double negativecount) {
		this.type = Type;
		this.entropyfinal = Entropy;
		this.leftchild = leftchild;
		this.rightchild = rightchild;
		this.attributeName = attr;
		this.value = value;
		this.positivecount = positivecount;
		this.negativecount = negativecount;
	}

	public Node(NodeType Type, Double Entropy, String attr, int value, double positivecount, double negativecount) {
		this.type = Type;
		this.entropyfinal = Entropy;
		this.attributeName = attr;
		this.value = value;
		this.positivecount = positivecount;
		this.negativecount = negativecount;
	}

	public Node getLeftchild() {
		return leftchild;
	}

	public Node getRightchild() {
		return rightchild;
	}

	public String toString() {
		if (type == NodeType.BRANCH) {
			return attributeName + value;
		} else {
			return " " + value;
		}
	}

}

public class Fxs161530 {

	public static void main(String[] args) {

		int L = Integer.parseInt(args[0]);
		int K = Integer.parseInt(args[1]);
		String trainingData = args[2];
		String testSet = args[3];
		String validationData = args[4];
		//String csvPath = "D:/Falak/Eclipse Workspace/MLAttempt1/datasets/data_sets1/training_set.csv";

		ArrayList<String[]> tdata = new ArrayList<String[]>(); //training data set
		ArrayList<String> usedAttrE = new ArrayList<String>(); // used attributes for Heuristics 1
		ArrayList<String> usedAttrV = new ArrayList<String>(); // used attributes for Heuristics 2


		String texample=""; // to read test data row
		BufferedReader br = null;

		try
		{

			br=new BufferedReader(new FileReader(trainingData));

			String attributes[] = br.readLine().split(","); // Header if the file, i.e first row having attribute Names
			//printing attributes
			System.out.println("Attributes : ");
			for(String attbr: attributes){
				System.out.print(attbr+", ");
			}System.out.println();
			
			while ((texample = br.readLine()) != null) {
				tdata.add(texample.split(","));
			}

			// Total examples
			int N = tdata.size();


			double pos=0,neg=0;

			for(String []record:tdata){

				//System.out.println(Integer.parseInt(record[record.length-1]));
				// for every row, check the value in the last column which represents C(i)
				if(Integer.parseInt(record[record.length-1])==0) 
					neg++;
				else pos++;

			}

			//convert to proportion
			pos = pos/N;
			neg=neg/N;

			//E(S) = entropy
			double entropy = (-pos)*(Math.log(pos)/Math.log(2))+(-neg)*(Math.log(neg)/Math.log(2));

			// build the tree based on Entropy
			Node rootE=null;
			rootE = buildTreeOnEntropy(tdata, usedAttrE, entropy, attributes, rootE, -1);


			// print the tree
			//	inOrder(rootE, " ");

			//building tree based on VI

			double vImpurity; // VI for S
			vImpurity = ((N-neg)*(N-pos))/(N^2);   // K0 = (N-pos) & K1 = (N-neg)
			Node rootV=null;
			rootV= buildTreeOnVarianceImpurity(tdata, usedAttrV, vImpurity, attributes, rootV, -1);

			// print the tree
			//	System.out.println("New Tree on VI ");
			//	inOrder(rootV, " ");

			String toPrint = args[5];

			if (toPrint.charAt(0) == 'Y'||toPrint.charAt(0) == 'y') {
				//Entropy
				System.out.println("Decision Tree by Entropy :\n");
				inOrder(rootE, " ");
				System.out.println("Accuracy by Entropy on test set is: " + AccuracyPercent(testSet, rootE));
				System.out.println("\nPost Pruning using Entropy\n");
				Node Dbest = postPruning(L, K, validationData, rootE);
				String x = " ";
				inOrder(Dbest, x);

				//Variance
				System.out.println("\nDecision Tree by Variance\n");
				inOrder(rootV, " ");
				System.out.println("Accuracy by Variance on test set is:" + AccuracyPercent(testSet, rootV));
				System.out.println();
				System.out.println("\nPost Pruning using Variance\n");
				Node Dbestf = postPruning(L, K, validationData, rootV);
				String y = " ";
				inOrder(Dbestf, y);

			} else { //Entropy
				System.out.println("\nDecision Tree by Entropy\n");
				System.out.println("\nAccuracy by Entropy is: " + AccuracyPercent(testSet, rootE));
				System.out.println("\nPost Pruning using Entropy\n");
				postPruning(L, K, validationData, rootE);

				//Variance
				System.out.println("\nDecision Tree by Variance\n");
				System.out.println("\nAccuracy by Variance is: " + AccuracyPercent(testSet, rootV));
				System.out.println("\nPost Pruning using Variance\n");
				postPruning(L, K, validationData, rootV);

			}


		}catch(Exception e){

			e.printStackTrace();
		}
		finally{
			try{
				br.close(); //preventing resource leak
			}catch(IOException ioe)
			{
				ioe.printStackTrace();
			}
		}
	}

	/* -------------------------------------------------------------------------------------------------------------------------*/

	public static Node buildTreeOnEntropy(ArrayList<String[]> records, ArrayList<String> usedAttrE, double entropy,
			String atr[], Node nodeE, int value) {



		ArrayList<String> newattributesE = new ArrayList<String>(usedAttrE);

		double entropyY = 0, entropyN = 0, Gain = 0, maxentropyY = 0, maxentropyN = 0, maxpos = 0, maxneg = 0;
		int iterator = 0, maxindex = 0;

		ArrayList<String[]> arrayleft = new ArrayList<String[]>();
		ArrayList<String[]> arrayright = new ArrayList<String[]>();

		Double max = Double.valueOf(Double.NEGATIVE_INFINITY);

		//The leaf node contains the final value
		if (newattributesE.size() == atr.length - 1 || records.isEmpty()) {
			return new Node(NodeType.LEAF, entropy, "" + value, value);
		} else if (entropy == 0) {
			return new Node(NodeType.LEAF, entropy, records.get(0)[records.get(0).length - 1], value); // return C(i) value in the node 

		} else {

			while (iterator != atr.length - 1) {
				double attrYclassY = 0, attrYclassN = 0, attrNclassY = 0, attrNclassN = 0, negative = 0, positive = 0;

				if (!newattributesE.contains(atr[iterator])) {
					for (String[] record : records) {
						int last = record.length - 1;

						//count pos and negative values in each ith attribute of each record
						if (Integer.parseInt(record[iterator]) == 0)
							negative++;
						else
							positive++;

						if (Integer.parseInt(record[last]) == 1) { //Class Yes
							if (Integer.parseInt(record[iterator]) == 1) { // Attribute is also Yes
								attrYclassY++;
							} else {
								attrNclassY++; //Class Yes, Attribute is No
							}
						} else {//Class No

							if (Integer.parseInt(record[iterator]) == 1) {// Attribute is  Yes
								attrYclassN++;
							} else {
								attrNclassN++; //Attribute is No
							}
						}
					}

					if (attrYclassY == 0 || attrYclassN == 0) {
						entropyY = 0;
					} else {
						entropyY = -((attrYclassY / (attrYclassY + attrYclassN))
								* ((Math.log(attrYclassY / (attrYclassY + attrYclassN))) / (Math.log(2))))
								- ((attrYclassN / (attrYclassY + attrYclassN))
										* ((Math.log(attrYclassN / (attrYclassN + attrYclassY))) / (Math.log(2))));
					}

					if (attrNclassY == 0 || attrNclassN == 0) {
						entropyN = 0;
					} else {
						entropyN = -((attrNclassY / (attrNclassY + attrNclassN))
								* ((Math.log(attrNclassY / (attrNclassY + attrNclassN))) / (Math.log(2))))
								- ((attrNclassN / (attrNclassY + attrNclassN))
										* ((Math.log(attrNclassN / (attrNclassY + attrNclassN))) / (Math.log(2))));
					}

					Gain = entropy - (((positive / (positive + negative)) * entropyY))
							- (((negative / (positive + negative)) * entropyN));

					if (max < Gain) {
						max = Gain;
						maxindex = iterator;
						maxentropyY = entropyY;
						maxentropyN = entropyN;
						maxpos = attrYclassY + attrNclassY; // = positive attribute i
						maxneg = attrYclassN + attrNclassN; // = negative attribute i
					}

				}

				iterator++; // consider next attribute

			}

			//create a branch at attribute with max entropy
			nodeE = new Node(NodeType.BRANCH, entropy, atr[maxindex], value, maxpos, maxneg);

			newattributesE.add(atr[maxindex]); // add atr[at max index] to used attributes

			//split the records based on the max entropy attribute


			for (String[] record : records) {
				if (Integer.parseInt(record[maxindex]) == 1) {
					arrayright.add(record);
				} else {
					arrayleft.add(record);
				}
			}

		}
		nodeE.leftchild = buildTreeOnEntropy(arrayleft, newattributesE, maxentropyN, atr, nodeE.leftchild, 0);
		nodeE.rightchild = buildTreeOnEntropy(arrayright, newattributesE, maxentropyY, atr, nodeE.rightchild, 1);
		return nodeE;
	}

	/* -------------------------------------------------------------------------------------------------------------------------*/

	//method to build tree on VI Heuristics
	public static Node buildTreeOnVarianceImpurity(ArrayList<String[]> records,ArrayList<String> usedAttrV, double impurity,
			String atr[], Node nodeV, int value) {

		ArrayList<String> newattributesV = new ArrayList<String>(usedAttrV);

		double impurityY = 0, impurityN = 0, Gain = 0, maxVIY = 0, maxVIN = 0, maxpos = 0, maxneg = 0;
		int iterator = 0, maxindex = 0;

		ArrayList<String[]> arrayleft = new ArrayList<String[]>();
		ArrayList<String[]> arrayright = new ArrayList<String[]>();

		Double max = Double.valueOf(Double.NEGATIVE_INFINITY);

		//The leaf node contains the final value
		if (newattributesV.size() == atr.length - 1 || records.isEmpty()) {
			return new Node(NodeType.LEAF, impurity, "" + value, value);
		} else if (impurity == 0) {
			return new Node(NodeType.LEAF, impurity, records.get(0)[records.get(0).length - 1]/*first record which is of type String[] and its 
			value at the last index which is at [record0.length-1] which is its C(i)*/, value); 

		} else {

			while (iterator != atr.length - 1) { // do for all attributes
				double attrYclassY = 0, attrYclassN = 0, attrNclassY = 0, attrNclassN = 0, negative = 0, positive = 0;

				if (!newattributesV.contains(atr[iterator])) { // attribute [i] has not been used before
					for (String[] record : records) {
						int last = record.length - 1;

						//count positive and negative values in each ith attribute of each record
						if (Integer.parseInt(record[iterator]) == 0)
							negative++;
						else
							positive++;

						if (Integer.parseInt(record[last]) == 1) { //Class Yes C(i)=1
							if (Integer.parseInt(record[iterator]) == 1) { // Attribute is also Yes
								attrYclassY++;
							} else {
								attrNclassY++; //Class Yes, Attribute is No
							}
						} else {//Class No

							if (Integer.parseInt(record[iterator]) == 1) {// Attribute is  Yes
								attrYclassN++;
							} else {
								attrNclassN++; //Attribute is No
							}
						}
					}

					if (attrYclassY == 0 || attrYclassN == 0) {
						impurityY = 0;
					} else {
						impurityY = (attrYclassY*attrYclassN)/((attrYclassY+attrYclassN)*(attrYclassY+attrYclassN));
					}

					if (attrNclassY == 0 || attrNclassN == 0) {
						impurityN = 0;
					} else {
						impurityN = (attrNclassY*attrNclassN)/((attrYclassY+attrYclassN)*(attrYclassY+attrYclassN));
					}

					Gain = impurity - (((positive / (positive + negative)) * impurityY))
							- (((negative / (positive + negative)) * impurityN));

					if (max < Gain) {
						max = Gain;
						maxindex = iterator;
						maxVIY = impurityY;
						maxVIN = impurityN;
						maxpos = attrYclassY + attrNclassY; // = positive attribute i
						maxneg = attrYclassN + attrNclassN; // = negative attribute i
					}

				}

				iterator++; // consider next attribute

			}

			//create a branch at attribute with max impurity
			nodeV = new Node(NodeType.BRANCH, impurity, atr[maxindex], value, maxpos, maxneg);

			newattributesV.add(atr[maxindex]); // add atr[at max index] to used attributes

			//split the records based on the max entropy attribute


			for (String[] record : records) {
				if (Integer.parseInt(record[maxindex]) == 1) {
					arrayright.add(record);
				} else {
					arrayleft.add(record);
				}
			}

		}
		nodeV.leftchild = buildTreeOnEntropy(arrayleft, newattributesV, maxVIN, atr, nodeV.leftchild, 0);
		nodeV.rightchild = buildTreeOnEntropy(arrayright, newattributesV, maxVIY, atr, nodeV.rightchild, 1);
		return nodeV;



	}

	/* -------------------------------------------------------------------------------------------------------------------------*/

	//Print the tree
	public static void inOrder(Node node, String s) {

		if (node == null)
			return;
		else if (node.type == NodeType.LEAF) 
		{
			System.out.println(": " + node.attributeName);
		} 

		else if (node.type == NodeType.BRANCH) {
			System.out.println();

			if (node.leftchild != null) {
				System.out.print(" | " + s);
				System.out.print(node.attributeName + " = " + node.leftchild.value);
				inOrder(node.getLeftchild(), s+ " | ");
			}
			if (node.rightchild != null) {
				System.out.print(" | " + s);
				System.out.print(node.attributeName + " = " + node.rightchild.value);
				inOrder(node.getRightchild(),s+" | ");
			}
		}
	}

	/* -------------------------------------------------------------------------------------------------------------------------*/

	//pruning the tree

	public static Node postPruning(int L, int K, String validationData, Node root){
		Node Dbest = root;
		int countnodes = 0;
		double accuracy = AccuracyPercent(validationData, root);
		System.out.println("\nAccuracy on Validation Set before Post Pruning is : " + accuracy);
		double accuracyBest = 0;
		for (int i = 0; i < L; i++) {
			Node Dtemp = null;
			Dtemp = copytree(root, Dtemp);
			int M = (1 + (int) (Math.random() * K));

			for (int j = 0; j < M; j++) {
				int P = 0;
				List<Node> nodeList = new ArrayList<>();
				nodeList = arraypattern(nodeList, Dtemp);
				countnodes = nodeList.size() - 2;

				P = (1 + (int) (Math.random() * countnodes));

				if (P != 0 && nodeList.size() >= 2) {
					Node replace = nodeList.get(P);
					replace.leftchild = null;
					replace.rightchild = null;
					replace.type = NodeType.LEAF;
					if (replace.positivecount > replace.negativecount) {
						replace.attributeName = "1";
					} else {
						replace.attributeName = "0";
					}
				}
			}
			accuracyBest = AccuracyPercent(validationData, Dtemp);
			if (accuracyBest > accuracy) {
				accuracy = accuracyBest;
				Dbest = Dtemp;	
			}
		}
		System.out.println("\nBest accuracy is : " + accuracy);
		return Dbest;
	}

	/* -------------------------------------------------------------------------------------------------------------------------*/

	public static Node copytree(Node finalroot, Node Dtemp) {
		if (finalroot.type == NodeType.LEAF) {
			Dtemp = new Node(finalroot.type, finalroot.entropyfinal, finalroot.attributeName, finalroot.value,
					finalroot.leftchild, finalroot.rightchild, 0, 0);
		} else {
			Dtemp = new Node(finalroot.type, finalroot.entropyfinal, finalroot.attributeName, finalroot.value, null,
					null, finalroot.positivecount, finalroot.negativecount);
			Dtemp.leftchild = copytree(finalroot.leftchild, Dtemp.leftchild);
			Dtemp.rightchild = copytree(finalroot.rightchild, Dtemp.rightchild);
		}
		return Dtemp;
	}
	/* -------------------------------------------------------------------------------------------------------------------------*/

	public static List<Node> arraypattern(List<Node> nodeList, Node finalroot) {

		if (finalroot != null && finalroot.type == NodeType.LEAF) {
			return nodeList;

		} else if (finalroot != null) {
			nodeList.add(finalroot);
			arraypattern(nodeList, finalroot.leftchild);
			arraypattern(nodeList, finalroot.rightchild);
		}
		return nodeList;
	}

	/* -------------------------------------------------------------------------------------------------------------------------*/
	public static double AccuracyPercent(String test, Node Rootcal) {
		FileInputStream  testset = null;
		BufferedReader  testfile = null;
		String attributes = " ";
		double accuracy = 0;
		int  totalsize = 0, cnt = 0;
		try {
			testset = new FileInputStream(test);
			testfile = new BufferedReader(new InputStreamReader(testset));
			attributes = testfile.readLine();
			String[] atrs = attributes.split(",");
			String[] test_set = new String[atrs.length];
			HashMap<String, Integer> attributeindex = new HashMap<String, Integer>();
			for (int i = 0; i < atrs.length; i++) {
				attributeindex.put(atrs[i], i);
			}
			String line1 = " ";
			int x = 0;
			while ((line1 = testfile.readLine()) != null) {

				test_set = line1.split(",");
				x = Accuracy(Rootcal, test_set, attributeindex);
				if (Integer.parseInt(test_set[atrs.length - 1]) == x) {
					cnt++;
				}
				totalsize++;
			}
			accuracy = ((double) cnt / totalsize) * 100;
		} catch (FileNotFoundException e) {
			System.out.println("File not found in given location.");
		} catch (IOException ex) {
			Logger.getLogger(BufferedReader.class.getName()).log(Level.SEVERE, null, ex);
		} finally {
			try {
				testfile.close();
			} catch (IOException ex) {
				Logger.getLogger(BufferedReader.class.getName()).log(Level.SEVERE, null, ex);
			}
		}
		return accuracy;
	}
	/* -------------------------------------------------------------------------------------------------------------------------*/	
	public static int Accuracy(Node node, String[] test_set, HashMap<String, Integer> attributeindex) {
		if (node.type == NodeType.LEAF) {
			return Integer.parseInt(node.attributeName);
		}
		String name = " ";
		name = node.attributeName;
		int index = attributeindex.get(name);

		if (Integer.parseInt(test_set[index]) == 0) {
			return Accuracy(node.leftchild, test_set, attributeindex);
		} else {
			return Accuracy(node.rightchild, test_set, attributeindex);
		}
	}
	
	/* -------------------------------------------------------------------------------------------------------------------------*/
}
