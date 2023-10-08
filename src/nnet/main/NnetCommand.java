package nnet.main;

import mxlib.excep.MxlibInvalidMatrixOp;
import nnet.main.cfg.NNetConfig;
import nnet.matrix.NNetMatrix;
import nnet.matrix.data.*;
import nnet.matrix.net.Network;

public class NnetCommand  {

	public static void main(String[] args) throws MxlibInvalidMatrixOp {
		
		NNetConfig cfg = NNetConfig.load("data/smallTest/NnetTest.json");

		try {
			Network n1 = new Network(cfg);
			n1.setDebugLevel(3);
			
			n1.train(cfg.trainingData(), 2);

			NnetBatchDataIntf tstdata = cfg.testData();
			

			n1.setDebugLevel(0);

			while (!tstdata.atEof()) {
				NNetBatch b = tstdata.nextBatch(1);
				NNetMatrix results = n1.predict(b.inData());
				
				results.print("Test Results");
				b.expectedResults().print("Expected");
			}
						
		
//			for (Double l : n1.getStats().lossArray)
//				System.out.println(l);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		

	}
}

