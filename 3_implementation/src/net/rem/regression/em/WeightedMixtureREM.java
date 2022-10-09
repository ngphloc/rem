/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.regression.em;

import java.rmi.RemoteException;

import net.hudup.core.data.DataConfig;
import net.hudup.core.logistic.LogUtil;
import net.rem.regression.LargeStatistics;
import net.rem.regression.em.ExchangedParameter.NormalDisParameter;

/**
 * This class implements the mixture regression model with weighting mechanism.
 * In fact, weights are added to EM coefficients. In this current implementation, these weights are response probabilities P(Z|X) and regressor probabilities P(X).
 * The method {@link #adjustMixtureParameters()} is responsible for calculating these weights. 
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class WeightedMixtureREM extends DefaultMixtureREM {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public WeightedMixtureREM() {

	}


	@Override
	protected boolean adjustMixtureParameters() throws RemoteException {
		super.adjustMixtureParameters();
		
		for (REMImpl rem : rems) {
			ExchangedParameter parameter = null;
			LargeStatistics stat = null;
			try {
				parameter = (ExchangedParameter)rem.getParameter();
				stat = (LargeStatistics) rem.expectation(parameter, this.data);
			} 
			catch (Exception e) {LogUtil.trace(e);}
			
			NormalDisParameter xNormalDisParameter = new NormalDisParameter(stat);
			parameter.setXNormalDisParameter(xNormalDisParameter);
		}
		
		return true;
	}


	@Override
	public String getName() {
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "mixrem_weighted";
	}

	
	@Override
	public DataConfig createDefaultConfig() {
		DataConfig config = super.createDefaultConfig();
		config.put(REMImpl.ESTIMATE_MODE_FIELD, REMImpl.REVERSIBLE);
		config.addInvisible(REMImpl.ESTIMATE_MODE_FIELD);
		return config;
	}
	
	
}
