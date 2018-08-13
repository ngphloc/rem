package net.hudup.regression.evaluate;

import net.hudup.Evaluator;
import net.hudup.core.alg.Alg;
import net.hudup.core.data.Profile;
import net.hudup.core.evaluate.testing.TestingEvaluator;
import net.hudup.regression.Regression;

/**
 * Evaluator for evaluating regression algorithms.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class RegressionEvaluator extends TestingEvaluator {

	
	/**
	 * Default constructor.
	 */
	public RegressionEvaluator() {
		// TODO Auto-generated constructor stub
		super();
	}

	
	@Override
	protected Object extractTestValue(Alg alg, Profile testingProfile) {
		// TODO Auto-generated method stub
		return ((Regression)alg).extractResponse(testingProfile);
	}

	
	@Override
	public boolean acceptAlg(Alg alg) {
		// TODO Auto-generated method stub
		return (alg instanceof Regression);
	}

	
	@Override
	public String getName() {
		// TODO Auto-generated method stub
		return "Regression Evaluator";
	}


	/**
	 * The main method to start evaluator.
	 * @param args The argument parameter of main method. It contains command line arguments.
	 * @throws Exception if there is any error.
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		String regressEvClassName = RegressionEvaluator.class.getName();
		System.setProperty("hudup_decimal_precision", "4");
		new Evaluator().run(new String[] {regressEvClassName});
	}

	
}
