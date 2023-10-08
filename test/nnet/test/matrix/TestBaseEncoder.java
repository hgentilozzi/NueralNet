package nnet.test.matrix;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import nnet.matrix.NNetMatrix;
import nnet.matrix.encorders.*;

class TestBaseEncoder {
	double epsilon = 0.000001d;

	@Test
	void testFit() {
		
		// One columns (select all)
		EncoderIntf enc = new BaseEncoder(); 
		assertFalse(enc.isFitted());

		NNetMatrix m = new NNetMatrix(new double[][] {{3},{2},{4}});
		enc.fit(m);
		assertTrue(enc.isFitted());
		assertTrue(Math.abs(3.0-enc.getMean()[0])<epsilon);
		assertTrue(Math.abs(0.6666666-enc.getVariance()[0])<epsilon);
		assertTrue(Math.abs(0.8164965-enc.getStdv()[0])<epsilon);
		
		// two columns (select all)
		enc = new BaseEncoder(); 
		m = new NNetMatrix(new double[][] {{3,5},{2,8},{4,12}});
		enc.fit(m);
		assertTrue(enc.isFitted());
		assertTrue(Math.abs(3.000000-enc.getMean()[0])<epsilon);
		assertTrue(Math.abs(8.333333-enc.getMean()[1])<epsilon);
		assertTrue(Math.abs(0.6666666-enc.getVariance()[0])<epsilon);
		assertTrue(Math.abs(8.2222222-enc.getVariance()[1])<epsilon);

		assertTrue(Math.abs(0.8164965-enc.getStdv()[0])<epsilon);
		assertTrue(Math.abs(2.8674417-enc.getStdv()[1])<epsilon);		
		
		// three columns (select first and last)
		enc = new BaseEncoder(); 
		m = new NNetMatrix(new double[][] {{3,4,5},{2,7,8},{4,13,12}});
		enc.fit(m,0,2);
		assertTrue(enc.isFitted());
		assertTrue(Math.abs(3.000000-enc.getMean()[0])<epsilon);
		assertTrue(Math.abs(8.333333-enc.getMean()[1])<epsilon);
		assertTrue(Math.abs(0.6666666-enc.getVariance()[0])<epsilon);
		assertTrue(Math.abs(8.2222222-enc.getVariance()[1])<epsilon);

		assertTrue(Math.abs(0.8164965-enc.getStdv()[0])<epsilon);
		assertTrue(Math.abs(2.8674417-enc.getStdv()[1])<epsilon);		

	}

	@Test
	void testTeansform() {
		EncoderIntf enc = new BaseEncoder(); 
		NNetMatrix m = new NNetMatrix(new double[][] {{3},{2},{4}});
		enc.fit(m);
				
		NNetMatrix r = enc.transform(m);
		assertEquals(0.0,r.get(0, 0));
		assertEquals(-1.224744871391589,r.get(1, 0));
		assertEquals(1.224744871391589,r.get(2, 0));
		
	}
	
}
