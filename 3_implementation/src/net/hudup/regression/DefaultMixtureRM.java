package net.hudup.regression;

import static net.hudup.em.AbstractEM.EM_EPSILON_FIELD;
import static net.hudup.em.AbstractEM.EM_MAX_ITERATION_FIELD;
import static net.hudup.em.EM.EM_DEFAULT_EPSILON;
import static net.hudup.em.EM.EM_MAX_ITERATION;
import static net.hudup.regression.em.DefaultMixtureREM.COMP_NUMBER_FIELD;
import static net.hudup.regression.em.DefaultMixtureREM.PREV_PARAMS_FIELD;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import net.hudup.core.alg.AbstractTestingAlg;
import net.hudup.core.alg.Alg;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.data.DataConfig;
import net.hudup.regression.em.DefaultMixtureREM;
import net.hudup.regression.em.ExchangedParameter;

/**
 * This class represents the default mixture regression model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class DefaultMixtureRM extends AbstractTestingAlg implements RM, DuplicatableAlg {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal regression model.
	 */
	protected DefaultMixtureREM mixREM = null;
	
	
	/**
	 * Name of maximum cluster number field.
	 */
	public final static String COMP_MAX_NUMBER_FIELD = "max_comp_number";

	
	/**
	 * Default maximum cluster number of cluster.
	 */
	public final static int COMP_MAX_NUMBER_DEFAULT = 10;

	
	@SuppressWarnings("unchecked")
	@Override
	public Object learn(Object... info) throws Exception {
		// TODO Auto-generated method stub
		DefaultMixtureREM prevMixREM = null;
		double prevFitness = -1;
		double threshold = getConfig().getAsReal(EM_EPSILON_FIELD);
		int maxK = getConfig().getAsInt(COMP_MAX_NUMBER_FIELD);
		maxK = maxK <= 0 ? Integer.MAX_VALUE : maxK;
		while (true) {
			DefaultMixtureREM mixREM = new DefaultMixtureREM();
			mixREM.getConfig().put(EM_EPSILON_FIELD, this.getConfig().get(EM_EPSILON_FIELD));
			mixREM.getConfig().put(EM_MAX_ITERATION_FIELD, this.getConfig().get(EM_MAX_ITERATION_FIELD));
			mixREM.getConfig().put(R_INDICES_FIELD, this.getConfig().get(R_INDICES_FIELD));
			mixREM.getConfig().put(COMP_NUMBER_FIELD, 1);
			
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
			if (fitness > prevFitness
					 && AbstractRM.notSatisfy(fitness, prevFitness, threshold)) {
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


	@Override
	public String getName() {
		// TODO Auto-generated method stub
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "cluster_mixrm";
	}

	
	@Override
	public Object execute(Object input) {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.execute(input);
		else
			return null;
	}


	@Override
	public Object extractResponse(Object input) {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.extractResponse(input);
		else
			return null;
	}


	@Override
	public Object getParameter() {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.getParameter();
		else
			return null;
	}


	@Override
	public String parameterToShownText(Object parameter, Object... info) {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.parameterToShownText(parameter, info);
		else
			return "";
	}


	@Override
	public String getDescription() {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.getDescription();
		else
			return "";
	}


	@Override
	public Alg newInstance() {
		// TODO Auto-generated method stub
		DefaultMixtureRM mixRegress = new DefaultMixtureRM();
		mixRegress.getConfig().putAll((DataConfig)this.getConfig().clone());
		return mixRegress;
	}

	
	@Override
	public void setName(String name) {
		// TODO Auto-generated method stub
		getConfig().put(DUPLICATED_ALG_NAME_FIELD, name);
	}


	@Override
	public DataConfig createDefaultConfig() {
		// TODO Auto-generated method stub
		DataConfig config = super.createDefaultConfig();
		config.put(EM_EPSILON_FIELD, EM_DEFAULT_EPSILON);
		config.put(EM_MAX_ITERATION_FIELD, EM_MAX_ITERATION);
		config.put(R_INDICES_FIELD, R_INDICES_DEFAULT);
		config.put(COMP_MAX_NUMBER_FIELD, COMP_MAX_NUMBER_DEFAULT);
		
		config.addReadOnly(DUPLICATED_ALG_NAME_FIELD);
		return config;
	}

	

}
