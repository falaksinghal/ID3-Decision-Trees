/**
 * Implement the decision tree learning algorithm. 
 * The following two heuristics are used for selecting the next attribute -
 * 1. Information gain heuristic
 * 2. Variance impurity heuristic. 
 * Implementation of post pruning algorithm to reduce the complexity of final classifier and
 * improve the accuracy of prediction and target concept by significantly reducing overfitting of the training data-set.
 *  Author : Falak Singhal (fxs@161530@utdallas.edu)
 */



package classes;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;

class processData{
	public Integer[][] createMatrix(String csvPath){

		ArrayList<String> data = new ArrayList<String>()	;
		String line="";


		try(BufferedReader br = new BufferedReader(new FileReader(csvPath))){
			while ((line = br.readLine()) != null) {
				data.add(line);
			}

		}catch (Exception e) {
			e.printStackTrace();
		}

		String[] header = data.get(0).split(",");
		data.remove(0);

		//total datasets 
		//System.out.println(data.size());
		//total attributes 
		//System.out.println(header.length);

		//Matrix Dimentions
		int totalRows=data.size();
		int totalColumns=header.length-1;

		Integer matrix[][] = new Integer[data.size()][header.length];

		for(int rows=0;rows<totalRows;rows++){   // rows 0 to 599


			String [] dataLine= data.get(rows).split(","); //
			//System.out.println(dataLine[4]);
			try{
				for(int col=0;col<totalColumns+1;col++){           // column 0 to 20 (includes C(i))              
					matrix[rows][col] = Integer.parseInt(dataLine[col]);

				}}catch (Exception e) {
					e.printStackTrace();
				}
		}

		//print matrix [0-599][0-20] Last column [index 20] is C(i) printed here
		for(int i=0;i<data.size();i++){
			System.out.print("Row : "+i+ "  ");
			for(int j=0;j<header.length;j++){
				System.out.print(matrix[i][j]+ " ");
			}
			System.out.println();
		}

		return (matrix);
	}
	
	
	public int[] calculatePosNeg(Integer mat[][], int colIndex){
		
		int pos=0,neg=0;
		int sum=0;
		
		int temp[]=new int[mat.length];
		//extract column
		for(int i=0;i<mat.length;i++){
			temp[i]=mat[i][colIndex];
		}
		for(int j=0;j<temp.length;j++)
		{sum+=temp[j];}
		
		pos=sum;
		neg=mat.length-sum;
		
		int val[]= new int[2];
		val[0]=pos;
		val[1]=neg;
		return val;
	}

}




public class ReadSplit {
	public static void main(String[] args) {

		String csvTrain = "D:/Eclipse Workspace/MLAttempt1/datasets/data_sets1/training_set.csv";
		processData process=new processData();
		Integer matrix[][]=process.createMatrix(csvTrain);
		
		int c[]=new int[matrix.length];
		//600 C(i) values
		for(int i=0;i<matrix.length;i++){
			c[i] = matrix[i][20];
		}
		
		//print C(i)
				for (int i=0;i<matrix.length;i++){
					System.out.print(c[i]+ " ");
				}
	
		int conceptSum[]=process.calculatePosNeg(matrix, 20);
		System.out.println("Pos :"+ conceptSum[0]);
				
		
	
	}
	
	
	
	
	
	
	
}
