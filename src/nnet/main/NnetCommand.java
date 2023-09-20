package nnet.main;

import nnet.exception.*;
import nnet.matrix.NNetMatrix;
import nnet.matrix.acvt.*;
import nnet.matrix.data.*;
import nnet.matrix.net.*;

public class NnetCommand  {

	public static void main(String[] args) throws NNetInvalidMatrixOp {
		NNetBatchDataFile batchDB = new NNetBatchDataFile("data/OneRowBatchDB.csv");	
		
		
		try {
			Network n1 = new Network(new ActivationFunctionSigmoid(),3,2,1);
			n1.setDebugLevel(1);
			
			n1.train(batchDB, 1);
			
			//System.out.println("Training stats: " + n1.getStats().toString());
			
			batchDB.reset();
			NNetBatch b = batchDB.nextBatch();
			
			NNetMatrix results = n1.predict(b.inData());
			
			results.print("Test Results");
			b.expectedResults().print("Expected");
			
			
		} catch (Exception e) {
			System.err.println(e);
		}
		
		

	}
}
