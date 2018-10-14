package net.hudup.temp;

import static net.hudup.em.AbstractEM.EM_EPSILON_FIELD;
import static net.hudup.em.AbstractEM.EM_MAX_ITERATION_FIELD;
import static net.hudup.regression.em.DefaultMixtureREM.COMP_NUMBER_FIELD;
import static net.hudup.regression.em.DefaultMixtureREM.PREV_PARAMS_FIELD;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import net.hudup.core.Util;
import net.hudup.core.alg.Alg;
import net.hudup.core.data.DataConfig;
import net.hudup.regression.DefaultMixtureRM;
import net.hudup.regression.em.DefaultMixtureREM;
import net.hudup.regression.em.ExchangedParameter;

public class DefaultMixtureRM2 extends DefaultMixtureRM {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@SuppressWarnings("unchecked")
	@Override
	public Object learn(Object... info) throws Exception {
		// TODO Auto-generated method stub
		DefaultMixtureREM prevMixREM = null;
		double prevFitness = -1;
		int maxK = getConfig().getAsInt(COMP_MAX_NUMBER_FIELD);
		maxK = maxK <= 0 ? Integer.MAX_VALUE : maxK;
		while (true) {
			DefaultMixtureREM mixREM = createInternalRM();
			
			if (prevMixREM != null) {
				List<ExchangedParameter> prevParameters = ExchangedParameter.clone((List<ExchangedParameter>)prevMixREM.getParameter());
				if (prevParameters instanceof Serializable)
					mixREM.getConfig().put(PREV_PARAMS_FIELD, (Serializable)prevParameters);
				else {
					ArrayList<ExchangedParameter> tempParameters = new ArrayList<>();
					tempParameters.addAll(prevParameters);
					mixREM.getConfig().put(PREV_PARAMS_FIELD, tempParameters);
				}
				mixREM.getConfig().put(COMP_NUMBER_FIELD, prevParameters.size() + 1);
			}
			if (prevMixREM == null)
				mixREM.setup(this.sample);
			else
				mixREM.setup(prevMixREM);
			
			// Breaking if zero alpha or zero coefficient.
			List<ExchangedParameter> parameters = (List<ExchangedParameter>)mixREM.getParameter();
			if (parameters == null || parameters.size() == 0 || parameters.size() > maxK) {
				mixREM.unsetup();
				break;
			}
			boolean breakhere = false;
			for (ExchangedParameter parameter : parameters) {
				if (parameter.getCoeff() == 0 || parameter.isNullAlpha()) {
					breakhere = true;
					break;
				}
			}
			if (breakhere) {
				mixREM.unsetup();
				break;
			}
			
			double fitness = mixREM.getFitness();
			if (Util.isUsed(fitness)
					&& fitness > prevFitness) {
				prevFitness = fitness;
				if (prevMixREM != null)
					prevMixREM.unsetup();
				prevMixREM = mixREM;
				
				if (((List<ExchangedParameter>)prevMixREM.getParameter()).size() >= maxK)
					break;
			}
			else {
				mixREM.unsetup();
				break;
			}
		}
		
		if (prevMixREM != null)
			prevMixREM.unsetup();
		this.mixREM = prevMixREM;
		return prevMixREM;
	}


	/**
	 * Creating internal regression model.
	 * @return internal regression model.
	 */
	protected DefaultMixtureREM createInternalRM() {
		DefaultMixtureREM2 mixREM = new DefaultMixtureREM2();
		mixREM.getConfig().put(EM_EPSILON_FIELD, this.getConfig().get(EM_EPSILON_FIELD));
		mixREM.getConfig().put(EM_MAX_ITERATION_FIELD, this.getConfig().get(EM_MAX_ITERATION_FIELD));
		mixREM.getConfig().put(R_INDICES_FIELD, this.getConfig().get(R_INDICES_FIELD));
		mixREM.getConfig().put(COMP_NUMBER_FIELD, 1);
		
		return mixREM;
	}
	
	
	@Override
	public String getName() {
		// TODO Auto-generated method stub
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "cluster_mixrm2";
	}


	@Override
	public Alg newInstance() {
		// TODO Auto-generated method stub
		DefaultMixtureRM2 mixRegress = new DefaultMixtureRM2();
		mixRegress.getConfig().putAll((DataConfig)this.getConfig().clone());
		return mixRegress;
	}

	
	@Override
	public void setName(String name) {
		// TODO Auto-generated method stub
		getConfig().put(DUPLICATED_ALG_NAME_FIELD, name);
	}


}
