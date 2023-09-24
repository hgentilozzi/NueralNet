package nnet.test.matrix;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import mxlib.excep.MxlibInvalidMatrixOp;
import nnet.matrix.NNetMatrix;
import nnet.matrix.acvt.ActivationFunction;
import nnet.matrix.net.*;

class TestLayer {
	
	@BeforeAll
	static void init() {
		NNetParameters.getInstance().setActivationFunction(ActivationFunction.RELU);
		NNetParameters.getInstance().setBatchSize(1);
		NNetParameters.getInstance().setLearningRate(0.001);		
	}
	
	@Test
	void testInputLayer() throws MxlibInvalidMatrixOp {
		// One row batch
		double[][] data = new double[][] {{1,2}};
		NNetMatrix x = new NNetMatrix(data);
		Layer l = new Layer(LayerType.INPUT_LAYER,2);
		l.setInputData(x);
		
		assertEquals(0, l.getLoss());
		assertEquals(1, l.getBatchSize());
		assertEquals(2, l.getNumNodes());
		assertArrayEquals(data[0],l.getOutputValues().getData()[0]);

		// two row batch
		data = new double[][] {{1,2},{3,4}};
		x = new NNetMatrix(data);
		l = new Layer(LayerType.INPUT_LAYER,2);
		l.setInputData(x);
		
		assertEquals(0, l.getLoss());
		assertEquals(2, l.getBatchSize());
		assertEquals(2, l.getNumNodes());
		assertArrayEquals(data[0],l.getOutputValues().getData()[0]);
		assertArrayEquals(data[1],l.getOutputValues().getData()[1]);

	}
	
	@Test
	void testOutputLayer() throws MxlibInvalidMatrixOp {
		// One row batch input layer
		double[][] idata = new double[][] {{1}};
		NNetMatrix im = new NNetMatrix(idata);
		Layer il = new Layer(LayerType.INPUT_LAYER,1);
		il.setInputData(im);
		
		// one node output layer
		double[][] odata = new double[][] {{1}};
		NNetMatrix om = new NNetMatrix(odata);
		Layer ol = new Layer(LayerType.OUTPUT_LAYER,1);
		ol.setExpectedData(om);
		
		il.setOutputLayer(ol);
		ol.setInputLayer(il);
		
		il.init();
		ol.init();
		
		double[][] wdata = new double[][] {{0.5}};
		NNetMatrix w1 = new NNetMatrix(wdata);
		ol.setWeight(w1);


		assertEquals(1, il.getBatchSize());
		assertEquals(1, il.getNumNodes());
		
		il.feedForward();
		ol.feedForward();
		assertArrayEquals((new double[][] {{0.5}})[0],ol.getOutputValues().getData()[0]);
				
		ol.backPropigation();

	}
	
	
	@Test
	void testHiddenLayer() throws MxlibInvalidMatrixOp {
		// One row batch input layer
		double[][] idata = new double[][] {{1}};
		NNetMatrix im = new NNetMatrix(idata);
		Layer il = new Layer(LayerType.INPUT_LAYER,1);
		il.setInputData(im);
		
		// one node output layer
		double[][] odata = new double[][] {{0}};
		NNetMatrix om = new NNetMatrix(odata);
		Layer ol = new Layer(LayerType.OUTPUT_LAYER,1);
		ol.setExpectedData(om);

		
		// two node hidden layer
		Layer hl = new Layer(LayerType.HIDDEN_LAYER,2,NNetParameters.getInstance().getActivationFunction());
		ol.setExpectedData(om);

		
		il.setOutputLayer(hl);
		hl.setInputLayer(il);
		hl.setOutputLayer(ol);
		ol.setInputLayer(hl);
		
		il.init();
		hl.init();
		ol.init();
		
		assertEquals(1, il.getBatchSize());
		assertEquals(1, il.getNumNodes());
		
		il.feedForward();
		hl.feedForward();
		ol.feedForward();

		
		ol.backPropigation();
		hl.backPropigation();
		il.backPropigation();

	}
	
	
	
	
	

}
