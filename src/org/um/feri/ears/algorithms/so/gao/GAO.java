package org.um.feri.ears.algorithms.so.gao;

import org.um.feri.ears.algorithms.AlgorithmInfo;
import org.um.feri.ears.algorithms.Author;
import org.um.feri.ears.algorithms.NumberAlgorithm;
import org.um.feri.ears.problems.*;
import org.um.feri.ears.util.annotation.AlgorithmParameter;
import org.um.feri.ears.util.random.PredefinedRandom;
import org.um.feri.ears.util.random.RNG;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class GAO extends NumberAlgorithm {

    // Custom random number generator state
    static class RandomState {
        long seed;
        long current;

        RandomState(long seed) {
            this.seed = seed;
            this.current = seed;
        }

        // Custom random number generator (same algorithm as MATLAB)
        public double customRand() {
            long a = 1664525L;
            long c = 1013904223L;
            long m = 1L << 32; // 2^32

            current = (a * current + c) % m;
            return (double) current / (double) m;
        }

        public int customRandInt(int minVal, int maxVal) {
            double r = this.customRand();
            return minVal + (int) Math.floor(r * (maxVal - minVal + 1));
        }
    }

    @AlgorithmParameter(name = "population size")
    private int popSize;
    public boolean isDebug = false;
    private ArrayList<NumberSolution<Double>> population;
    private NumberSolution<Double> bestSolution;
    private RandomState randomState;
    public GAO() {
        this(30);
    }

    public GAO(int popSize) {
        super();
        randomState = new RandomState(42);
        this.popSize = popSize;
        this.isDebug = isDebug;
        au = new Author("matej", "matej.habjanic@student.um.si");
        ai = new AlgorithmInfo("GAO", "Giant Armadillo Optimization",
                "@article{Armadillos,\n" +
                        "  title = {Giant Armadillo Optimization: A New Bio-Inspired Metaheuristic Algorithm for Solving Optimization Problems},\n" +
                        "  volume = {8},\n" +
                        "  ISSN = {2313-7673},\n" +
                        "  url = {http://dx.doi.org/10.3390/biomimetics8080619},\n" +
                        "  DOI = {10.3390/biomimetics8080619},\n" +
                        "  number = {8},\n" +
                        "  journal = {Biomimetics},\n" +
                        "  publisher = {MDPI AG},\n" +
                        "  author = {Alsayyed,  Omar and Hamadneh,  Tareq and Al-Tarawneh,  Hassan and Alqudah,  Mohammad and Gochhait,  Saikat and Leonova,  Irina and Malik,  Om Parkash and Dehghani,  Mohammad},\n" +
                        "  year = {2023},\n" +
                        "  month = dec,\n" +
                        "  pages = {619}\n" +
                        "}"
        );
    }

    @Override
    public NumberSolution<Double> execute(Task<NumberSolution<Double>, DoubleProblem> task) throws StopCriterionException {
        this.task = task;
        initPopulation();

        int maxIterations = 0;
        List<Double> lowerBounds = task.problem.getLowerLimit();
        List<Double> upperBounds = task.problem.getUpperLimit();

        if (task.getStopCriterion() == StopCriterion.ITERATIONS) {
            maxIterations = task.getMaxIterations();
        }

        if (task.getStopCriterion() == StopCriterion.EVALUATIONS) {
            maxIterations = (task.getMaxEvaluations() - popSize) / (popSize * 2);
        }

        for (int iteration = 1; iteration <= maxIterations; iteration++) {
            /*System.out.println("------------");
            for(int k=0;k<population.size();k++) {
                System.out.println(Arrays.toString(population.get(k).getVariables().toArray()));
            }
            System.out.println("------------");
            System.out.println();
            System.out.println();*/

            if (task.isStopCriterion()) {
                break;
            }

            for (int i = 0; i < popSize; i++) {
                if (task.isStopCriterion()) {
                    break;
                }

                NumberSolution<Double> currentAgent = population.get(i);
                double currentFitness = currentAgent.getEval();

                // PHASE 1: Attack on termite mounds (Exploration)
                NumberSolution<Double> termiteMound = selectTermiteMound(currentFitness);

                NumberSolution<Double> newSolution1 = generateNewSolutionPhase1(
                        currentAgent, termiteMound, lowerBounds, upperBounds);

                if (task.isStopCriterion()) break;
                task.eval(newSolution1);

                if (newSolution1.getEval() < currentFitness) {
                    population.set(i, newSolution1);
                    currentFitness = newSolution1.getEval();
                    currentAgent = newSolution1;
                    if (currentFitness < bestSolution.getEval()) {
                        bestSolution = new NumberSolution<>(newSolution1);
                        if (isDebug) {
                            System.out.println("New best solution found: " + Arrays.toString(bestSolution.getVariables().toArray()) +
                                    " with fitness: " + bestSolution.getEval());
                        }
                    }
                }

                // PHASE 2: Digging in termite mounds (Exploitation)
                NumberSolution<Double> newSolution2 = generateNewSolutionPhase2(
                        currentAgent, lowerBounds, upperBounds, iteration);

                if (task.isStopCriterion()) break;
                task.eval(newSolution2);

                if (newSolution2.getEval() <= currentFitness) {
                    population.set(i, newSolution2);

                    if (newSolution2.getEval() < bestSolution.getEval()) {
                        bestSolution = new NumberSolution<>(newSolution2);
                        if (isDebug) {
                            System.out.println("New best solution found: " + Arrays.toString(bestSolution.getVariables().toArray()) +
                                    " with fitness: " + bestSolution.getEval());
                        }
                    }
                }
            }
            iteration++;
        }

        return bestSolution;
    }

    private NumberSolution<Double> selectTermiteMound(double currentFitness) {
        ArrayList<NumberSolution<Double>> betterSolutions = new ArrayList<>();

        for (NumberSolution<Double> solution : population) {
            if (solution.getEval() < currentFitness) {
                betterSolutions.add(solution);
            }
        }

        if (betterSolutions.isEmpty()) {
            return bestSolution;
        }

        int randomIndex = (int) (RNG.nextUniform()/*randomState.customRand()*/ * betterSolutions.size());
        return betterSolutions.get(randomIndex);
    }

    private NumberSolution<Double> generateNewSolutionPhase1(NumberSolution<Double> current,
                                                             NumberSolution<Double> termiteMound,
                                                             List<Double> lb, List<Double> ub
    ) {
        List<Double> newPosition = new ArrayList<>();

        for (int j = 0; j < current.getVariables().size(); j++) {
            double I = RNG.nextInt(1, 2);//randomState.customRandInt(1, 2);
            double r = RNG.nextUniform();//randomState.customRand();
            double newValue = current.getValue(j) + r * (termiteMound.getValue(j) - I * current.getValue(j));
            newValue = Math.max(newValue, lb.get(j));
            newValue = Math.min(newValue, ub.get(j));
            newPosition.add(newValue);
        }

        return new NumberSolution<Double>(newPosition);
    }

    private NumberSolution<Double> generateNewSolutionPhase2(NumberSolution<Double> current,
                                                             List<Double> lb, List<Double> ub,
                                                             int iteration) {
        List<Double> newPosition = new ArrayList<>();
        double t = iteration + 1;

        for (int j = 0; j < current.getVariables().size(); j++) {
            double newValue = current.getValue(j) + (1 - 2 * RNG.nextUniform()/*randomState.customRand()*/) * (ub.get(j) - lb.get(j)) / t;
            newValue = Math.max(newValue, lb.get(j) / t);
            newValue = Math.min(newValue, ub.get(j) / t);
            newPosition.add(newValue);
        }

        return new NumberSolution<>(newPosition);
    }

    @Override
    public void resetToDefaultsBeforeNewRun() {
        population = null;
        bestSolution = null;
    }

    private void initPopulation() throws StopCriterionException {
        population = new ArrayList<>();
        bestSolution = null;

        List<Double> lowerBounds = task.problem.getLowerLimit();
        List<Double> upperBounds = task.problem.getUpperLimit();
        int dimension = task.problem.getNumberOfDimensions();

        for (int i = 0; i < popSize; i++) {
            if (task.isStopCriterion()) {
                break;
            }
            /*ArrayList<Double> initialPosition = new ArrayList<>();
            for(int d = 0; d < dimension; d++) {
                double val = this.randomState.customRand() * (upperBounds.get(d) - lowerBounds.get(d)) + lowerBounds.get(d);
                initialPosition.add(val);
            }

            NumberSolution<Double> newSol = new NumberSolution<>();
            newSol.setVariables(initialPosition);
            task.eval(newSol);
            population.add(newSol);*/

            NumberSolution<Double> newSolution = task.generateRandomEvaluatedSolution();
            population.add(newSolution);

            // Initialize best solution
            if (bestSolution == null || newSolution.getEval() < bestSolution.getEval()) {
                bestSolution = new NumberSolution<>(newSolution);
            }
        }
    }
}

/*
function[Best_score,Best_pos,GAO_curve]=GAO(SearchAgents,Max_iterations,lowerbound,upperbound,dimension,fitness)
    % Initialize custom random generator
    rng_state = initRandomState(12345);

    lowerbound=ones(1,dimension).*(lowerbound);                              % Lower limit for variables
    upperbound=ones(1,dimension).*(upperbound);                              % Upper limit for variables

    %% INITIALIZATION
    for i=1:dimension
        X(i,:) = lowerbound(i) + customRand(rng_state).*(upperbound(i) - lowerbound(i));                          % Initial population
    end

    for i =1:SearchAgents
        L=X(i,:);
        fit(i)=fitness(L);
    end
    %%

    for t=1:Max_iterations  % algorithm iteration

        %%  update: BEST proposed solution
        [Fbest , blocation]=min(fit);

        if t==1
            xbest=X(blocation,:);                                           % Optimal location
            fbest=Fbest;                                           % The optimization objective function
        elseif Fbest<fbest
            fbest=Fbest;
            xbest=X(blocation,:);
        end
        %%
        %%
        %disp(X)        % simple, human-readable
        X               % no semicolon prints the variable with its name

        for i=1:SearchAgents
            %%
                %% Phase 1: Attack on termite mounds (exploration phase)
                TM_location=find(fit<fit(i));% based on Eq(4)
                if size (TM_location,2)==0
                    STM=xbest;
                else
                    % Replace randperm with custom random integer
                    K = customRandInt(rng_state, 1, size(TM_location,2));
                    STM=X(K,:);
                end
                I=round(1+customRand(rng_state));
                X_new_P1=X( i,: ) + customRand(rng_state).* ( STM - I.*X( i,: ) ) ;%Eq(5)
                X_new_P1 = max(X_new_P1,lowerbound);X_new_P1 = min(X_new_P1,upperbound);

                % update position based on Eq (6)
                L=X_new_P1;
                fit_new_P1=fitness(L);
                if fit_new_P1<fit(i)
                    X(i,:) = X_new_P1;
                    fit(i) = fit_new_P1;
                end
                %% End Phase 1

                %% Phase 2: Digging in termite mounds (exploitation phase)
                X_new_P2=X(i,:)+ (1-2*customRand(rng_state)).*(upperbound-lowerbound)./t; % Eq(7)
                X_new_P2= max(X_new_P2,lowerbound/t);X_new_P2 = min(X_new_P2,upperbound/t);

                % Updating X_i using (8)
                L=X_new_P2;
                f_new = fitness(L);
                if f_new <= fit (i)
                    X(i,:) = X_new_P2;
                    fit (i)=f_new;
                    if f_new<fbest
                        xbest=X_new_P2;
                        fbest=f_new;
                    end
                end
                %%

            %%

        end % for i=1:SearchAgents
        %%

        best_so_far(t)=fbest;
        average(t) = mean (fit);

    end
    Best_score=fbest;
    Best_pos=xbest;
    GAO_curve=best_so_far;
end

% Custom random number generator functions
function r = customRand(state)
    % Linear Congruential Generator for reproducible random numbers
    a = 1664525;
    c = 1013904223;
    m = 2^32;

    state.current = mod(a * state.current + c, m);
    r = double(state.current) / double(m);
end

function r_int = customRandInt(state, min_val, max_val)
    % Generate random integer between min_val and max_val (inclusive)
    r = customRand(state);
    r_int = min_val + floor(r * (max_val - min_val + 1));
end

function state = initRandomState(seed)
    % Initialize random state with given seed
    state = struct('seed', seed, 'current', seed);
end
 */