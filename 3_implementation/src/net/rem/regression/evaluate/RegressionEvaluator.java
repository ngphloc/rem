/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.regression.evaluate;

import java.io.Serializable;
import java.rmi.RemoteException;

import net.hudup.Evaluator;
import net.hudup.core.alg.Alg;
import net.hudup.core.data.Profile;
import net.hudup.core.evaluate.execute.ExecuteEvaluator;
import net.hudup.core.logistic.LogUtil;
import net.rem.regression.RM;

/**
 * Evaluator for evaluating regression algorithms.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class RegressionEvaluator extends ExecuteEvaluator {

	
	/**
	 * Default serial version UID.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public RegressionEvaluator() {
		super();
	}

	
	@Override
	protected Serializable extractTestValue(Alg alg, Profile testingProfile) {
		try {
			return (Serializable) ((RM)alg).extractResponseValue(testingProfile);
		} catch (Exception e) {LogUtil.trace(e);}
		
		return null;
	}

	
	@Override
	public boolean acceptAlg(Alg alg) throws RemoteException {
		return (alg != null) && (alg instanceof RM);
	}


	@Override
	public String getName() {
		return "regress";
	}


	/**
	 * The main method to start evaluator.
	 * @param args The argument parameter of main method. It contains command line arguments.
	 * @throws Exception if there is any error.
	 */
	public static void main(String[] args) throws Exception {
		String regressEvClassName = RegressionEvaluator.class.getName();
		new Evaluator().run(new String[] {regressEvClassName});
	}

	
}
