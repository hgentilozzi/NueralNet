package nnet.main;

import nnet.exception.*;
import nnet.matrix.NNetPerceptron;
import nnet.matrix.acvt.ActivationFunctionReLU;
import nnet.matrix.net.*;

public class NnetCommand  {

	public static void main(String[] args) throws NNetInvalidMatrixOp {
		NNetPerceptron i_data  = new NNetPerceptron(new double[][] {{0.9}});				
		NNetPerceptron o_data  = new NNetPerceptron(new double[][] {{0.3}});	
			
		try {
			Network n1 = new Network(new ActivationFunctionReLU(),1,2,1);
			
			n1.train(i_data, o_data, 10000);
			
			System.out.println("Training stats: " + n1.getStats().toString());
			
			NNetPerceptron results = n1.predict(i_data);
			
			results.print("Results");
			
			
		} catch (NNetInvalidNetwork e) {
			System.err.println(e);
			return;
		}
		
		

	}
}
