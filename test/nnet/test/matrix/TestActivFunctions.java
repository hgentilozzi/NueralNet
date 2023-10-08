package nnet.test.matrix;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import nnet.matrix.NNetMatrix;
import nnet.matrix.acvt.*;

class TestActivFunctions {
	private ActivationFunctionReLU afReLU = new ActivationFunctionReLU();
	private ActivationFunctionTanh afTanh = new ActivationFunctionTanh();
	private ActivationFunctionSigmoid afSigm = new ActivationFunctionSigmoid();
	private ActivationFunctionSoftMax sftMax = new ActivationFunctionSoftMax();

	@Test
	void testReLU() {
		afReLU.setLeakValue(0.0);
		NNetMatrix m = new NNetMatrix(new double[][] {{-1.0,0.0,2.6}}); 
		NNetMatrix r = afReLU.f(m);
		
		assertEquals(-0.0,r.get(0, 0));
		assertEquals(0,r.get(0, 1));
		assertEquals(2.6,r.get(0, 2));
		
		m = new NNetMatrix(new double[][] {{-1.0,0.0,2.6}});
		NNetMatrix g = afReLU.grad(m);
		assertEquals(0,g.get(0, 0));
		assertEquals(1,g.get(0, 1));
		assertEquals(1,g.get(0, 2));
		
	}

	@Test
	void testReLULeaky() {
		afReLU.setLeakValue(0.01);
		NNetMatrix m = new NNetMatrix(new double[][] {{-2.0,0.0,2.6}}); 
		NNetMatrix r = afReLU.f(m);
		
		assertEquals(-0.02,r.get(0, 0));
		assertEquals(0,r.get(0, 1));
		assertEquals(2.6,r.get(0, 2));
		
		m = new NNetMatrix(new double[][] {{-6.0,0.0,2.6}});
		NNetMatrix g = afReLU.grad(m);
		assertEquals(0.01,g.get(0, 0));
		assertEquals(1.0,g.get(0, 1));
		assertEquals(1.0,g.get(0, 2));
		
	}
	@Test
	void testTanh() {
		double[][] data = new double[][] {{-1.0,0.0,1}};
		NNetMatrix m = new NNetMatrix(data); 
		NNetMatrix r = afTanh.f(m);
		
		assertEquals(-0.7615941559557649,r.get(0, 0));
		assertEquals(0,r.get(0, 1));
		assertEquals(0.7615941559557649,r.get(0, 2));
	}
	
	@Test
	void testSigm() {
		NNetMatrix m = new NNetMatrix(new double[][] {{-2.0,0.0,2}}); 
		NNetMatrix r = afSigm.f(m);
		
		assertEquals(0.11920292202211755,r.get(0, 0));
		assertEquals(0.5,r.get(0, 1));
		assertEquals(0.8807970779778823,r.get(0, 2));

		m = new NNetMatrix(new double[][] {{-2.0,0.0,2}}); 
		NNetMatrix g = afSigm.grad(m);
		assertEquals(0.1049935854035065,g.get(0, 0));
		assertEquals(0.25,g.get(0, 1));
		assertEquals(0.10499358540350662,g.get(0, 2));

	}

	
	@Test
	void testSoftMax() {
		double[][] data = new double[][] {{-2.0,0.0,2}};
		NNetMatrix m = new NNetMatrix(data); 
		NNetMatrix r = sftMax.f(m);
		
		assertEquals(0.01587623997646677,r.get(0, 0));
		assertEquals(0.11731042782619837,r.get(0, 1));
		assertEquals(0.8668133321973349,r.get(0, 2));
		
		NNetMatrix g = sftMax.grad(r);
		assertEquals(0.34851885870495364,g.get(0, 0));
		assertEquals(0.3737849316504316,g.get(0, 1));
		assertEquals(0.4986037370127617,g.get(0, 2));
	}

}
