package nnet.test.matrix;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import nnet.exception.NNetInvalidMatrixOp;
import nnet.matrix.NNetPerceptron;
import nnet.matrix.net.*;

class TestLayer {
	//private ActivationFunctionReLU afReLU = new ActivationFunctionReLU();

	@Test
	void testInputLayer() throws NNetInvalidMatrixOp {
		// One row batch
		double[][] data = new double[][] {{1,2}};
		NNetPerceptron x = new NNetPerceptron(data);
		Layer l = new Layer(LayerType.INPUT_LAYER,2);
		l.setInputData(x);
		
		assertEquals(0, l.getLoss());
		assertEquals(1, l.getBatchSize());
		assertEquals(2, l.getNumNodes());
		assertArrayEquals(data[0],l.getOutputValues().getData()[0]);

		// two row batch
		data = new double[][] {{1,2},{3,4}};
		x = new NNetPerceptron(data);
		l = new Layer(LayerType.INPUT_LAYER,2);
		l.setInputData(x);
		
		assertEquals(0, l.getLoss());
		assertEquals(2, l.getBatchSize());
		assertEquals(2, l.getNumNodes());
		assertArrayEquals(data[0],l.getOutputValues().getData()[0]);
		assertArrayEquals(data[1],l.getOutputValues().getData()[1]);

	}
	
	@Test
	void testOutputLayer() throws NNetInvalidMatrixOp {
		// One row batch input layer
		double[][] idata = new double[][] {{1}};
		NNetPerceptron im = new NNetPerceptron(idata);
		Layer il = new Layer(LayerType.INPUT_LAYER,1);
		il.setInputData(im);
		
		// one node output layer
		double[][] odata = new double[][] {{0}};
		NNetPerceptron om = new NNetPerceptron(odata);
		Layer ol = new Layer(LayerType.OUTPUT_LAYER,1);
		ol.setExpectedData(om);
		
		il.setOutputLayer(ol);
		ol.setInputLayer(il);
		
		il.init();
		ol.init();
		
		double[][] wdata = new double[][] {{0.5}};
		NNetPerceptron w1 = new NNetPerceptron(wdata);
		ol.setWeight(w1);


		assertEquals(1, il.getBatchSize());
		assertEquals(1, il.getNumNodes());
		
		il.feedForward();
		assertArrayEquals((new double[][] {{0.5}})[0],ol.getOutputValues().getData()[0]);
				
		ol.backPropigation();

	}
	
	
	@Test
	void testHiddenLayer() throws NNetInvalidMatrixOp {
		// One row batch input layer
		double[][] idata = new double[][] {{1}};
		NNetPerceptron im = new NNetPerceptron(idata);
		Layer il = new Layer(LayerType.INPUT_LAYER,1);
		il.setInputData(im);
		
		// one node output layer
		double[][] odata = new double[][] {{0}};
		NNetPerceptron om = new NNetPerceptron(odata);
		Layer ol = new Layer(LayerType.OUTPUT_LAYER,1);
		ol.setExpectedData(om);

		
		// two node hidden layer
		Layer hl = new Layer(LayerType.HIDDEN_LAYER,2);
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
