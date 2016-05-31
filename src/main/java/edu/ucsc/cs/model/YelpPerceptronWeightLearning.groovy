package edu.ucsc.cs.model;

import java.text.DecimalFormat;

import edu.umd.cs.psl.application.inference.LazyMPEInference
import edu.umd.cs.psl.application.inference.MPEInference;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood;
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin;
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.DatabasePopulator;
import edu.umd.cs.psl.database.Partition;
import edu.umd.cs.psl.database.ReadOnlyDatabase;
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.memory.MemoryFullInferenceResult
import edu.umd.cs.psl.groovy.PSLModel;
import edu.umd.cs.psl.groovy.PredicateConstraint;
import edu.umd.cs.psl.groovy.SetComparison;
import edu.umd.cs.psl.model.argument.ArgumentType;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.argument.UniqueID;
import edu.umd.cs.psl.model.argument.Variable;
import edu.umd.cs.psl.model.atom.GroundAtom;
import edu.umd.cs.psl.model.function.ExternalFunction;
import edu.umd.cs.psl.ui.functions.textsimilarity.*
import edu.umd.cs.psl.ui.loading.InserterUtils;
import edu.umd.cs.psl.util.database.Queries;



ConfigManager cm = ConfigManager.getManager()
ConfigBundle config = cm.getBundle("basic-example")

def defaultPath = System.getProperty("java.io.tmpdir")
String dbpath = config.getString("dbpath", defaultPath + File.separator + "rec_sys_yelp")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config)

sq = true;	//when sq=false then the potentials are linear while when sq is true the potentials are squared

Double totalRMSE = 0.0;
Double totalMAE = 0.0;

int num_folds = 5;

double[] average_rating = new double[num_folds];
average_rating[0]= 0.763756	//these are the outputs of the awk script for the 80 percent of the training file
average_rating[1]= 0.765197
average_rating[2]= 0.764513
average_rating[3]= 0.764182
average_rating[4]= 0.764629


for(int fold=1;fold<=num_folds;fold++){

PSLModel m = new PSLModel(this, data)

//DEFINITION OF THE MODEL
//general predicates
m.add predicate: "user", 			types: [ArgumentType.UniqueID]
m.add predicate: "item",			types: [ArgumentType.UniqueID]
m.add predicate: "rating",			types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "rated",			types: [ArgumentType.UniqueID, ArgumentType.UniqueID] //this is used in the blocking mechanism

//item similarities
m.add predicate: "sim_pearson_items",	types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "sim_cosine_items",   types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "sim_jaccard_items",   types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "sim_adjcos_items",	types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

//user similarities
m.add predicate: "sim_pearson_users",   types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "sim_cosine_users",   types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "sim_jaccard_users",   types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

//similar users/items computed over the users*factors and items*factors low dimensional matrices produced by Matrix Factorization
m.add predicate: "sim_mf_cosine_users",		types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "sim_mf_euclidean_users",	types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

m.add predicate: "sim_mf_cosine_items", 		types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "sim_mf_euclidean_items",		types: [ArgumentType.UniqueID, ArgumentType.UniqueID]


//predictions
m.add predicate: "sgd_rating", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "item_pearson_rating", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "bpmf_rating", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "pmf_rating", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

//average priors (users and items)
m.add predicate: "avg_user_rating",	types: [ArgumentType.UniqueID]
m.add predicate: "avg_item_rating", types: [ArgumentType.UniqueID]

//social info
m.add predicate: "users_are_friends",	types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "user_is_influential", types: [ArgumentType.UniqueID]

//demo-like info
m.add predicate: "user_is_active", types: [ArgumentType.UniqueID]

//content info
m.add predicate: "item_belongs_to_category", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "sim_content_items_jaccard", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

//overall prior
m.add predicate: "rating_prior",        types: [ArgumentType.UniqueID]


//RULES

//SIMILARITIES

//ITEMS
//pearson items
m.add rule :  ((rated(U,I1) & rated(U,I2) & rating(U,I1) & sim_pearson_items(I1,I2)) >> rating(U,I2)), weight: 1, squared:false;

//cosine items
m.add rule :  ((rated(U,I1) & rated(U,I2) & rating(U,I1) & sim_cosine_items(I1,I2)) >> rating(U,I2)), weight: 1, squared:false;

//adj cosine items
m.add rule :  ((rated(U,I1) & rated(U,I2) & rating(U,I1) & sim_adjcos_items(I1,I2)) >> rating(U,I2)), weight: 1, squared:false;

//USERS
//pearson users
m.add rule :  ((rated(U1,I) & rated(U2,I) & rating(U1,I) & sim_pearson_users(U1,U2)) >> rating(U2,I)), weight: 1, squared:false;

//cosine users
m.add rule :  ((rated(U1,I) & rated(U2,I) & rating(U1,I) & sim_cosine_users(U1,U2)) >> rating(U2,I)), weight: 1, squared:false;

//low dimension space similarities
//USERS
m.add rule :  ((user(U1) & user(U2) & item(I) & rating(U1,I) & rated(U1,I) &rated(U2,I) & sim_mf_cosine_users(U1,U2))
			>> rating(U2,I)), weight : 1, squared:false;

m.add rule :  ((user(U1) & user(U2) & item(I) & rating(U1,I) & rated(U1,I) &rated(U2,I) & sim_mf_euclidean_users(U1,U2))
			>> rating(U2,I)), weight : 1, squared:false;

//ITEMS		
//similar items get similar ratings: similarity is computed using the cosine metric over the matrix items*factors produced by the Non Negative MF
m.add rule :  ((user(U) & item(I1) & item(I2) & rating(U,I1) & rated(U,I1) & rated(U,I2) & sim_mf_cosine_items(I1,I2))
			>> rating(U,I2)), weight: 1, squared:false;

m.add rule :  ((user(U) & item(I1) & item(I2) & rating(U,I1) & rated(U,I1) & rated(U,I2) & sim_mf_euclidean_items(I1,I2))
			>> rating(U,I2)), weight: 1, squared:false;



//PREDICTIONS
//SGD Predictions
m.add rule: ((sgd_rating(U,I)) >> rating(U,I)), weight:1, squared:sq;
m.add rule: ((rating(U,I)) >> sgd_rating(U,I)), weight:1, squared:sq;

//item based pearson Predictions
m.add rule: ((item_pearson_rating(U,I)) >> rating(U,I)), weight:1, squared:sq;
m.add rule: ((rating(U,I)) >> item_pearson_rating(U,I)), weight:1, squared:sq;

//BPMF predictions
m.add rule: ((bpmf_rating(U,I)) >> rating(U,I)), weight:1, squared:sq;
m.add rule: ((rating(U,I)) >> bpmf_rating(U,I)), weight:1, squared:sq;

//AVERAGE PRIORS
m.add rule : ((user(U) & item(I) & rated(U,I) & avg_user_rating(U)) >> rating(U,I)), weight: 1, squared:sq;
m.add rule : ((user(U) & item(I) & rated(U,I) & rating(U,I)) >> avg_user_rating(U)), weight: 1, squared:sq;

//items tend to get ratings towards their average
m.add rule : ((user(U) & item(I) & rated(U,I) & avg_item_rating(I)) >> rating(U,I)), weight: 1, squared:sq;
m.add rule : ((user(U) & item(I) & rated(U,I) & rating(U,I)) >> avg_item_rating(I)), weight: 1, squared:sq;

//SOCIAL RULES
//friendships
m.add rule: ((rated(U1,I) & rated(U2,I) & users_are_friends(U1,U2) & rating(U1,I)) >> rating(U2,I)), weight: 4, squared:false;


//CONTENT RULES

m.add rule :  ((rated(U,I1) & rated(U,I2) & rating(U,I1) & sim_content_items_jaccard(I1,I2)) >> rating(U,I2)), weight: 1, squared:false;


// Two-sided overall prior
UniqueID constant = data.getUniqueID(0)
m.add rule: ( user(U) & item(I) & rated(U,I) & rating_prior(constant) ) >> rating(U,I), weight: 1, squared:sq;
m.add rule: ( user(I) & item(I) & rated(U,I) & rating(U,I) ) >> rating_prior(constant), weight: 1, squared:sq;

println m;



//keep track of the time
TimeNow = new Date();
println "start time is " + TimeNow

int num_similar_items = 50;
int num_similar_users = 50;
int ext = 60;
def dir = 'data'+java.io.File.separator+'Yelp5Folds'+java.io.File.separator + 'scottsdale'+ java.io.File.separator;
def w_dir = "weight_learning_PSL" + java.io.File.separator;
String graphlab = "graphlab_results" + java.io.File.separator;
String pmfbpmf = "pmf-bpmf-predictions" + java.io.File.separator;
String cf_items = "cf-" + num_similar_items + java.io.File.separator;
String cf_users = "cf-" + num_similar_users + java.io.File.separator;
String bpmf_sim = "bpmf-similarities" + java.io.File.separator;


	//we put in the same partition things that are observed
	def evidencePartition = new Partition(0 + fold * num_folds);	     // observed data for weight learning
	def targetPartition = new Partition(1 + fold * num_folds);		     // unobserved data for weight learning
	Partition trueDataPartition = new Partition(2 + fold * num_folds);  // train set for inference
	def evidencePartition2 = new Partition(3 + fold * num_folds);		 // test set for inference 
	def targetPartition2 = new Partition(4 + fold * num_folds);
	
	
	
	def insert = data.getInserter(user, evidencePartition)
	InserterUtils.loadDelimitedData(insert, dir + "users");
	
	insert = data.getInserter(item, evidencePartition)
	InserterUtils.loadDelimitedData(insert, dir + "items");
	
	insert = data.getInserter(rated, evidencePartition)
	InserterUtils.loadDelimitedData(insert, dir + w_dir + "rated." + fold);

	insert = data.getInserter(rating, evidencePartition);
	InserterUtils.loadDelimitedDataTruth(insert, dir + w_dir + "90pct.f" + fold + ".train");
	
	//simialrities: sim items pearson
	insert = data.getInserter(sim_pearson_items, evidencePartition)
	InserterUtils.loadDelimitedDataTruth(insert, dir + cf_items + "cf.f" + fold + ".train.pearson.sim.items.keep.top." + num_similar_items + ".boolean");

	//similarities: sim items cosine
	insert = data.getInserter(sim_cosine_items, evidencePartition)
	InserterUtils.loadDelimitedDataTruth(insert, dir + cf_items + "cf.f" + fold + ".train.cosine.sim.items.keep.top." + num_similar_items + ".boolean");

	//similarities: sim items adjcos
	insert = data.getInserter(sim_adjcos_items, evidencePartition)
	InserterUtils.loadDelimitedDataTruth(insert, dir + cf_items + "cf.f" + fold + ".train.adjcos.sim.items.keep.equal.1.boolean");

	//simialrities: sim users pearson
	insert = data.getInserter(sim_pearson_users, evidencePartition)
	InserterUtils.loadDelimitedDataTruth(insert, dir + cf_users + "cf.f" + fold + ".train.pearson.sim.users.keep.top." + num_similar_users + ".boolean");

	//similarities: sim users cosine
	insert = data.getInserter(sim_cosine_users, evidencePartition)
	InserterUtils.loadDelimitedDataTruth(insert, dir + cf_users + "cf.f" + fold + ".train.cosine.sim.users.keep.top." + num_similar_users + ".boolean");

	insert = data.getInserter(sim_mf_cosine_users, evidencePartition)
	InserterUtils.loadDelimitedDataTruth(insert, dir + bpmf_sim + "bpmf.u" + fold + ".base.cosine.sim.users.keep.top."+num_similar_users +".boolean");
	
	insert = data.getInserter(sim_mf_euclidean_users, evidencePartition)
	InserterUtils.loadDelimitedDataTruth(insert, dir + bpmf_sim + "bpmf.u" + fold + ".base.euclidean.sim.users.keep.top."+num_similar_users+".boolean");
	
	insert = data.getInserter(sim_mf_cosine_items, evidencePartition)
	InserterUtils.loadDelimitedDataTruth(insert, dir + bpmf_sim + "bpmf.u" + fold + ".base.cosine.sim.items.keep.top."+num_similar_items+".boolean");
	
	insert = data.getInserter(sim_mf_euclidean_items, evidencePartition)
	InserterUtils.loadDelimitedDataTruth(insert, dir + bpmf_sim + "bpmf.u" + fold + ".base.euclidean.sim.items.keep.top."+num_similar_items+".boolean");	

	//SGD predictions
	insert = data.getInserter(sgd_rating, evidencePartition)
	InserterUtils.loadDelimitedDataTruth(insert, dir + w_dir + "sgd.predictions.10.factors.fold." + fold + ".clean");

	//BPMF predictions
	insert = data.getInserter(bpmf_rating, evidencePartition)
	InserterUtils.loadDelimitedDataTruth(insert, dir + w_dir + "bpmf.predictions.10.factors.fold." + fold);

	//Item based pearson predictions
	insert = data.getInserter(item_pearson_rating, evidencePartition)
	InserterUtils.loadDelimitedDataTruth(insert, dir + w_dir + "10pct.item.based.pearson.top.1.predictions.fold." + fold + ".clean");

	insert = data.getInserter(avg_user_rating, evidencePartition)
	InserterUtils.loadDelimitedDataTruth(insert, dir + w_dir + "f" + fold + ".train.avg.users.ratings");
	
	insert = data.getInserter(avg_item_rating, evidencePartition)
	InserterUtils.loadDelimitedDataTruth(insert, dir + w_dir + "f" + fold + ".train.avg.items.ratings");

	//social info
	insert = data.getInserter(users_are_friends, evidencePartition)
	InserterUtils.loadDelimitedData(insert, dir + "friends.txt");

	insert = data.getInserter(sim_content_items_jaccard, evidencePartition)
	InserterUtils.loadDelimitedDataTruth(insert, dir + "sim.items.content.jaccard.above.0.5.boolean");

	// to predict
	insert = data.getInserter(rating, targetPartition)
	InserterUtils.loadDelimitedData(insert, dir + w_dir + "topredict." + fold);
	
	//total prior
	data.getInserter(rating_prior, evidencePartition).insertValue(average_rating[fold-1], constant);
	
	//target partition
	Database db = data.getDatabase(targetPartition, [user, item, rated,
		sim_mf_cosine_users, sim_mf_euclidean_users, 
		sim_mf_cosine_items, sim_mf_euclidean_items,
		sim_pearson_items, sim_cosine_items, sim_adjcos_items, sim_jaccard_items,
		sim_pearson_users, sim_cosine_users, sim_jaccard_users,
		sgd_rating, item_pearson_rating,
		bpmf_rating, pmf_rating,
		avg_item_rating, avg_user_rating,
		users_are_friends, user_is_influential, user_is_active,
		item_belongs_to_category, sim_content_items_jaccard] as Set, evidencePartition);
	
	
	//learn the weights
	/* learn the weights from data. For that, we need to have some
	 * evidence data from which we can learn. In our example, that means we need to
	 * specify ratings, which we now load into another partition.
	 */
	insert = data.getInserter(rating, trueDataPartition)
	InserterUtils.loadDelimitedDataTruth(insert, dir + w_dir + "10pct.f" + fold + ".train");
	/* Now, we can learn the weights.
	 * We first open a database which contains all the target atoms as observations.
	 * We then combine this database with the original database to learn.
	 */

	println "Start weight learning..."
	Database trueDataDB = data.getDatabase(trueDataPartition, [rating] as Set);
	
	MaxLikelihoodMPE weightLearning = new MaxLikelihoodMPE(m, db, trueDataDB, config);
	
	weightLearning.learn();
	weightLearning.close();
	
	//print the new model
	println ""
	println "Learned model:"
	println m
	db.close();	//close this db as we will not use it again
	
	//keep track of the time
	TimeNow = new Date();
	println "time after weight learning is " + TimeNow
	
	
	//perform inference with weight learning
	//we put in the same partition things that are observed
	
	insert = data.getInserter(user, evidencePartition2)
	InserterUtils.loadDelimitedData(insert, dir + "users");
	
	insert = data.getInserter(item, evidencePartition2)
	InserterUtils.loadDelimitedData(insert, dir + "items");
	
	insert = data.getInserter(rated, evidencePartition2)
	InserterUtils.loadDelimitedData(insert, dir + "rated");

	insert = data.getInserter(rating, evidencePartition2);
	InserterUtils.loadDelimitedDataTruth(insert, dir + "f" + fold + ".yelp.train.clean");
	
	//simialrities: sim items pearson
	insert = data.getInserter(sim_pearson_items, evidencePartition2)
	InserterUtils.loadDelimitedDataTruth(insert, dir + cf_items + "cf.f" + fold + ".train.pearson.sim.items.keep.top." + num_similar_items + ".boolean");

	//similarities: sim items cosine
	insert = data.getInserter(sim_cosine_items, evidencePartition2)
	InserterUtils.loadDelimitedDataTruth(insert, dir + cf_items + "cf.f" + fold + ".train.cosine.sim.items.keep.top." + num_similar_items + ".boolean");

	//similarities: sim items adjcos
	insert = data.getInserter(sim_adjcos_items, evidencePartition2)
	InserterUtils.loadDelimitedDataTruth(insert, dir + cf_items + "cf.f" + fold + ".train.adjcos.sim.items.keep.equal.1.boolean");

	//simialrities: sim users pearson
	insert = data.getInserter(sim_pearson_users, evidencePartition2)
	InserterUtils.loadDelimitedDataTruth(insert, dir + cf_users + "cf.f" + fold + ".train.pearson.sim.users.keep.top." + num_similar_users + ".boolean");

	//similarities: sim users cosine
	insert = data.getInserter(sim_cosine_users, evidencePartition2)
	InserterUtils.loadDelimitedDataTruth(insert, dir + cf_users + "cf.f" + fold + ".train.cosine.sim.users.keep.top." + num_similar_users + ".boolean");

	insert = data.getInserter(sim_mf_cosine_users, evidencePartition2)
	InserterUtils.loadDelimitedDataTruth(insert, dir + bpmf_sim + "bpmf.u" + fold + ".base.cosine.sim.users.keep.top."+num_similar_users+".boolean");
	
	insert = data.getInserter(sim_mf_euclidean_users, evidencePartition2)
	InserterUtils.loadDelimitedDataTruth(insert, dir + bpmf_sim + "bpmf.u" + fold + ".base.euclidean.sim.users.keep.top."+num_similar_users+".boolean");
	
	insert = data.getInserter(sim_mf_cosine_items, evidencePartition2)
	InserterUtils.loadDelimitedDataTruth(insert, dir + bpmf_sim + "bpmf.u" + fold + ".base.cosine.sim.items.keep.top."+num_similar_items+".boolean");
	
	insert = data.getInserter(sim_mf_euclidean_items, evidencePartition2)
	InserterUtils.loadDelimitedDataTruth(insert, dir + bpmf_sim + "bpmf.u" + fold + ".base.euclidean.sim.items.keep.top."+num_similar_items+".boolean");

	//SGD predictions
	insert = data.getInserter(sgd_rating, evidencePartition2)
	InserterUtils.loadDelimitedDataTruth(insert, dir + graphlab + "sgd.predictions.50.factors.fold." + fold);

	//BPMF predictions
	insert = data.getInserter(bpmf_rating, evidencePartition2)
	InserterUtils.loadDelimitedDataTruth(insert, dir + pmfbpmf + "bpmf.predictions.10.factors.fold." + fold);

	//Item based pearson predictions
	insert = data.getInserter(item_pearson_rating, evidencePartition2)
	InserterUtils.loadDelimitedDataTruth(insert, dir + graphlab + "item.based.pearson.top.1.predictions.fold." + fold);

	insert = data.getInserter(avg_user_rating, evidencePartition2)
	InserterUtils.loadDelimitedDataTruth(insert, dir + "f" + fold + ".train.avg.users.ratings");
	
	insert = data.getInserter(avg_item_rating, evidencePartition2)
	InserterUtils.loadDelimitedDataTruth(insert, dir + "f" + fold + ".train.avg.items.ratings");

	//social info
	insert = data.getInserter(users_are_friends, evidencePartition2)
	InserterUtils.loadDelimitedData(insert, dir + "friends.txt");

	//content info
	insert = data.getInserter(sim_content_items_jaccard, evidencePartition2)
	InserterUtils.loadDelimitedDataTruth(insert, dir + "sim.items.content.jaccard.above.0.5.boolean");

	// to predict
	insert = data.getInserter(rating, targetPartition2)
	InserterUtils.loadDelimitedData(insert, dir + "topredict." + fold);
	
	average_rating[0]= 0.764001	//these are the outputs of the awk script
	average_rating[1]= 0.764720
	average_rating[2]= 0.764770
	average_rating[3]= 0.764127
	average_rating[4]= 0.764622
		
	//average prior
	data.getInserter(rating_prior, evidencePartition2).insertValue(average_rating[fold-1], constant);

	//target partition
	Database db2 = data.getDatabase(targetPartition2, [user, item, rated,
		sim_mf_cosine_users, sim_mf_euclidean_users, 
		sim_mf_cosine_items, sim_mf_euclidean_items,
		sim_pearson_items, sim_cosine_items, sim_adjcos_items, sim_jaccard_items,
		sim_pearson_users, sim_cosine_users, sim_jaccard_users,
		sgd_rating, item_pearson_rating,
		bpmf_rating, pmf_rating,
		avg_item_rating, avg_user_rating,
		users_are_friends, user_is_influential, user_is_active,
		item_belongs_to_category, sim_content_items_jaccard] as Set, evidencePartition2);
	
	
	//perform MPEInference
	//create the target partition
	
	
	//run MPE inference with learned weights
	MPEInference inferenceApp = new MPEInference(m, db2, config);
	MemoryFullInferenceResult inf_result = inferenceApp.mpeInference();
	
	if(inf_result.getTotalWeightedIncompatibility()!=null)
		println "[DEBUG inference]: Incompatibility = " + inf_result.getTotalWeightedIncompatibility()
	if(inf_result.getInfeasibilityNorm()!=null)
		println "[DEBUG inference]: Infeasibility = " + inf_result.getInfeasibilityNorm()
	
	inferenceApp.close();
	
	
	//keep track of the time
	TimeNow = new Date();
	println "after the inference time is " + TimeNow
	
	
	//call the garbage collector - just in case!
	System.gc();
	
	//Compute the RMSE
	HashMap<String, HashMap<String, Double>> users_items_ratings_labels = new HashMap<String, HashMap<String, Double>>();
	def labels = new File(dir + "f" + fold + ".yelp.test.clean")
	def words, user, item, rating_value
	labels.eachLine {
		line ->
		words = line.split("\t")
		user=words[0].toString();
		item=words[1].toString();
		rating_value=words[2].toDouble();
		//user already exists
		if(users_items_ratings_labels.containsKey(user)){
			HashMap<String, Double> items_ratings = users_items_ratings_labels.get(user)
			items_ratings.put(item, rating_value)
		}
		else{	//first time to create an entry for this user
			HashMap<String, Double> items_ratings = new HashMap<String, Double>()
			items_ratings.put(item, rating_value)
			users_items_ratings_labels.put(user, items_ratings)
		}
	}
	
	println "Inference results with weights learned from perceptron algorithm:"
	Double RMSE = 0.0
	Double MAE = 0.0
	int n=0
	int number_of_higher_predictions = 0;
	for (GroundAtom atom : Queries.getAllAtoms(db2, rating)){
		user = atom.arguments[0].toString()
		item = atom.arguments[1].toString()
		rating_predicted = atom.getValue().toDouble()
		//search in the structure users_items_ratings_labels for the pair <user,item>
		//and if it does exist then compute the RMSE error
		if(users_items_ratings_labels.containsKey(user)){
			HashMap<String, Double> items_ratings = users_items_ratings_labels.get(user);
			if(items_ratings.containsKey(item)){
				rating_labeled = items_ratings.get(item);
				println "( " + user + "," + item + " ) = " + rating_predicted + "\t" + rating_labeled;
				RMSE += (rating_labeled-rating_predicted)*(rating_labeled-rating_predicted);
				MAE += Math.abs(rating_labeled-rating_predicted);
				if(rating_labeled < rating_predicted)
					number_of_higher_predictions++;
				//also count the number of ratings that the prediction is higher than the actual rating
				
				n++;
			}
		}
	}
	
	RMSE = Math.sqrt(RMSE/(1.0*n))
	totalRMSE += RMSE
	MAE = MAE/(1.0*n)
	totalMAE += MAE;
	per_higher_pred = (number_of_higher_predictions*1.0)/(1.0*n)
	println "number of higher predictions than the labeled " + number_of_higher_predictions
	println "percentage of higher predictions than the labeled " + per_higher_pred
	
	println "RMSE " + RMSE*5
	println "MAE " + MAE*5
	db2.close();

}//end folds

//compute the total RMSE and MAE

Double avgRMSE = totalRMSE/num_folds;
println "Avg RMSE = " + avgRMSE*5

Double avgMAE = totalMAE/num_folds;
println "Avg MAE = " + avgMAE*5

//keep track of the time
TimeNow = new Date();
println "end time is " + TimeNow


