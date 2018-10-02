package net.hudup.regression;

import static net.hudup.em.AbstractEM.EM_EPSILON_FIELD;
import static net.hudup.em.AbstractEM.EM_MAX_ITERATION_FIELD;
import static net.hudup.em.EM.EM_DEFAULT_EPSILON;
import static net.hudup.em.EM.EM_MAX_ITERATION;
import static net.hudup.regression.em.AbstractMixtureREM.COMP_EXECUTION_DEFAULT;
import static net.hudup.regression.em.AbstractMixtureREM.COMP_EXECUTION_FIELD;
import static net.hudup.regression.em.DefaultMixtureREM.COMP_NUMBER_FIELD;
import static net.hudup.regression.em.DefaultMixtureREM.PREV_PARAMS_FIELD;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JOptionPane;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.AbstractTestingAlg;
import net.hudup.core.alg.Alg;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.data.DataConfig;
import net.hudup.core.logistic.ui.UIUtil;
import net.hudup.regression.em.DefaultMixtureREM;
import net.hudup.regression.em.ExchangedParameter;
import net.hudup.regression.em.ui.graph.Graph;

/**
 * This class represents the default mixture regression model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class DefaultMixtureRM extends AbstractTestingAlg implements RM2, DuplicatableAlg {

	
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


	/**
	 * Creating internal regression model.
	 * @return internal regression model.
	 */
	protected DefaultMixtureREM createInternalRM() {
		DefaultMixtureREM mixREM = new DefaultMixtureREM();
		mixREM.getConfig().put(EM_EPSILON_FIELD, this.getConfig().get(EM_EPSILON_FIELD));
		mixREM.getConfig().put(EM_MAX_ITERATION_FIELD, this.getConfig().get(EM_MAX_ITERATION_FIELD));
		mixREM.getConfig().put(R_INDICES_FIELD, this.getConfig().get(R_INDICES_FIELD));
		mixREM.getConfig().put(COMP_NUMBER_FIELD, 1);
		mixREM.getConfig().put(COMP_EXECUTION_FIELD, this.getConfig().get(COMP_EXECUTION_FIELD));
		
		return mixREM;
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
	public LargeStatistics getLargeStatistics() {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.getLargeStatistics();
		else
			return null;
	}


	@Override
	public double executeByXStatistic(double[] xStatistic) {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.executeByXStatistic(xStatistic);
		else
			return Constants.UNUSED;
	}


	@Override
	public Object execute(Object input) {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.execute(input);
		else
			return Constants.UNUSED;
	}


	@Override
	public VarWrapper extractRegressor(int index) {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.extractRegressor(index);
		else
			return null;
	}


	@Override
	public List<VarWrapper> extractRegressors() {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.extractRegressors();
		else
			return Util.newList();
	}


	@Override
	public List<VarWrapper> extractSingleRegressors() {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.extractSingleRegressors();
		else
			return Util.newList();
	}


	@Override
	public double extractRegressorValue(Object input, int index) {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.extractRegressorValue(input, index);
		else
			return Constants.UNUSED;
	}


	@Override
	public VarWrapper extractResponse() {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.extractResponse();
		else
			return null;
	}


	@Override
	public Object extractResponseValue(Object input) {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.extractResponseValue(input);
		else
			return null;
	}


	@Override
	public List<Double> extractRegressorStatistic(VarWrapper regressor) {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.extractRegressorStatistic(regressor);
		else
			return Util.newList();
	}


	@Override
	public Object transformResponse(Object z, boolean inverse) {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.transformResponse(z, inverse);
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
	public synchronized void manifest() {
		// TODO Auto-generated method stub
		if (mixREM != null)
			mixREM.manifest();
		else
			JOptionPane.showMessageDialog(
				UIUtil.getFrameForComponent(null), 
				"Invalid regression model", 
				"Invalid regression model", 
				JOptionPane.ERROR_MESSAGE);
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
		config.put(COMP_EXECUTION_FIELD, COMP_EXECUTION_DEFAULT);
		
		config.addReadOnly(DUPLICATED_ALG_NAME_FIELD);
		return config;
	}


	@Override
	public Graph createRegressorGraph(VarWrapper regressor) {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.createRegressorGraph(regressor);
		else
			return null;
	}


	@Override
	public Graph createResponseGraph() {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.createResponseGraph();
		else
			return null;
	}


	@Override
	public Graph createErrorGraph() {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.createErrorGraph();
		else
			return null;
	}


	@Override
	public List<Graph> createResponseRalatedGraphs() {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.createResponseRalatedGraphs();
		else
			return Util.newList();
	}


	@Override
	public double calcVariance() {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.calcVariance();
		else
			return Constants.UNUSED;
	}


	@Override
	public double calcR() {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.calcR();
		else
			return Constants.UNUSED;
	}


	@Override
	public double[] calcError() {
		// TODO Auto-generated method stub
		if (mixREM != null)
			return mixREM.calcError();
		else
			return null;
	}

	

}
