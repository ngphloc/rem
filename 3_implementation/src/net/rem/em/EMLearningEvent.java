/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.em;

import java.io.Serializable;
import java.rmi.Remote;

import net.hudup.core.PluginStorage;
import net.hudup.core.alg.Alg;
import net.hudup.core.alg.SetupAlgEvent;
import net.hudup.core.data.Dataset;
import net.hudup.core.data.DatasetUtil;
import net.hudup.core.data.Exportable;
import net.hudup.core.data.Simplify;
import net.hudup.core.logistic.LogUtil;

/**
 * This class represents events when expectation maximization (EM) algorithm runs.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class EMLearningEvent extends SetupAlgEvent implements Simplify {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Current iteration.
	 */
	protected int currentIteration = 0;

	
	/**
	 * Sufficient statistics.
	 */
	protected Serializable currentStatistics = null;
	
	
	/**
	 * Current parameter.
	 */
	protected Serializable currentParameter = null;

	
	/**
	 * Estimated parameter.
	 */
	protected Serializable estimatedParameter = null;

	
	/**
	 * Constructor with some important parameters.
	 * @param em the EM algorithm as the source of this event. This EM algorithm is invalid in remote call because the source is transient variable.
	 * @param type event type.
	 * @param trainingDataset training dataset.
	 * @param currentIteration current iteration.
	 * @param maxIteration maximum iteration.
	 * @param currentStatistics current sufficient statistic.
	 * @param currentParameter current parameter.
	 * @param estimatedParameter estimated parameter of algorithm as setup result.
	 */
	public EMLearningEvent(EM em, Type type, Dataset trainingDataset,
			int currentIteration, int maxIteration, Serializable currentStatistics,
			Serializable currentParameter, Serializable estimatedParameter) {
		this(em, type, -1, trainingDataset, currentIteration, maxIteration, currentStatistics, currentParameter, estimatedParameter);
	}

	
	/**
	 * Constructor with some important parameters.
	 * @param em the EM algorithm as the source of this event. This EM algorithm is invalid in remote call because the source is transient variable.
	 * @param type event type.
	 * @param trainingDatasetId training dataset identifier.
	 * @param trainingDataset training dataset.
	 * @param currentIteration current iteration.
	 * @param maxIteration maximum iteration.
	 * @param currentStatistics current sufficient statistic.
	 * @param currentParameter current parameter.
	 * @param estimatedParameter estimated parameter of algorithm as setup result.
	 */
	public EMLearningEvent(EM em, Type type, int trainingDatasetId, Dataset trainingDataset,
			int currentIteration, int maxIteration, Serializable currentStatistics,
			Serializable currentParameter, Serializable estimatedParameter) {
		super(em, type, em.getName(), trainingDatasetId, trainingDataset, estimatedParameter, currentIteration, maxIteration);
		this.currentIteration = currentIteration;
		this.currentStatistics = currentStatistics;
		this.currentParameter = currentParameter; 
		this.estimatedParameter = estimatedParameter;
	}

	
	@Override
	public SetupAlgEvent transferForRemote() {
		EMLearningEvent evt = new EMLearningEvent(
				getEM(),
				this.type,
				this.trainingDatasetId,
				null,
				this.currentIteration, 
				this.progressTotalEstimated, 
				this.currentStatistics,
				this.currentParameter,
				this.estimatedParameter);
		if (this.trainingDataset == null) return evt;
		
		if (this.trainingDatasetId < 0)
			evt.trainingDatasetId = DatasetUtil.getDatasetId(getTrainingDataset());
		
		Dataset dataset = DatasetUtil.getMostInnerDataset2(this.trainingDataset);
		if ((dataset != null) && (dataset instanceof Exportable)) {
			Remote stub = null;
			try {
				stub = ((Exportable)dataset).getExportedStub();
			} catch (Exception e) {LogUtil.trace(e);}
			
			//It assures that only light remote dataset is transferred via network.
			if (stub != null) evt.trainingDataset = this.trainingDataset;
		}
		
		return evt;
	}


	/**
	 * Getting source as EM algorithm. This method cannot be called remotely because the source is transient variable.
	 * @return source as EM algorithm.
	 */
	private EM getEM() {
		Object source = getSource();
		if (source == null)
			return null;
		else if (source instanceof EM)
			return (EM)source;
		else
			return null;
	}

	
	@Override
	public String translate() {
		Alg alg = PluginStorage.getNormalAlgReg().query(getAlgName());
		if ((alg == null) || !(alg instanceof EM))
			return "";
		else
			return translate((EM)alg, false);
	}
	
	
	/**
	 * Translating this event into text with specified EM algorithm.
	 * @param em EM algorithm.
	 * @return translated text of this event.
	 */
	public String translate(EM em) {
		return translate(em, false);
	}
	
	
	/**
	 * Translate this event into text.
	 * @param em EM algorithm.
	 * @param showStatistic whether sufficient statistic is shown.
	 * This parameter is established because sufficient statistic often as a collection (fetcher) is very large.
	 * @return translated text from content of this event.
	 */
	protected String translate(EM em, boolean showStatistic) {
		if (em == null) return "";
		
		StringBuffer buffer = new StringBuffer();
		try { 
			buffer.append("At the " + currentIteration + " iteration");
			buffer.append(" of algorithm \"" + em.getName() + "\"");
			if (getTrainingDataset() != null) {
				String mainUnit = getTrainingDataset().getConfig().getMainUnit();
				String datasetName = getTrainingDataset().getConfig().getAsString(mainUnit);
				if (datasetName != null)
					buffer.append(" on training dataset \"" + datasetName + "\"");
			}

			if (currentStatistics != null && showStatistic) {
				buffer.append("\nCurrent statistic:");
				buffer.append("\n  " + currentStatistics.toString());
			}
			
			if (currentParameter != null) {
				buffer.append("\nCurrent parameter:");
				buffer.append("\n  " + em.parameterToShownText(currentParameter));
			}
			
			if (estimatedParameter != null) {
				buffer.append("\nEstimated parameter:");
				buffer.append("\n  " + em.parameterToShownText(estimatedParameter));
			}
		}
		catch (Throwable e) {
			LogUtil.trace(e);
		}
		
		return buffer.toString();
	}
	
	
	@Override
	public Object simplify() throws Exception {
		Dataset trainingDataset = DatasetUtil.getMostInnerDataset2(this.trainingDataset);
		if ((trainingDataset == null) && (this.trainingDataset instanceof Dataset))
			trainingDataset = (Dataset)this.trainingDataset;
		
		SetupAlgEvent evt = new SetupAlgEvent(
			Integer.valueOf(0), this.type, this.algName,
			this.trainingDatasetId,
			trainingDataset,
			this.setupResult != null ? this.setupResult.toString() : "",
			this.progressStep, progressTotalEstimated);
		
		return evt;
	}


}
