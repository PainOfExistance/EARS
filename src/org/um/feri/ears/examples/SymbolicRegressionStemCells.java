package org.um.feri.ears.examples;

import org.um.feri.ears.algorithms.GPAlgorithm;
import org.um.feri.ears.algorithms.gp.ElitismGPAlgorithm;
import org.um.feri.ears.individual.representations.gp.Node;
import org.um.feri.ears.individual.representations.gp.Target;
import org.um.feri.ears.individual.representations.gp.symbolic.regression.*;
import org.um.feri.ears.problems.StopCriterion;
import org.um.feri.ears.problems.StopCriterionException;
import org.um.feri.ears.problems.Task;
import org.um.feri.ears.problems.gp.ProgramProblem;
import org.um.feri.ears.problems.gp.ProgramSolution;
import org.um.feri.ears.problems.gp.SymbolicRegressionProblem;

import java.io.BufferedReader;
import java.io.FileReader;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SymbolicRegressionStemCells {

    public static void main(String[] args) {
        List<Class<? extends Node>> baseFunctionNodeTypes = Arrays.asList(
                AddNode.class,
                SubNode.class,
                MulNode.class,
                DivNode.class
        );

        List<Class<? extends Node>> baseTerminalNodeTypes = Arrays.asList(
                ConstNode.class,
                VarNode.class
        );

        ArrayList<Target> training = new ArrayList<>();

        List<String> headers = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader("D:\\FERI\\Faks\\Magisterij\\2.letnik\\1.semester\\ER\\final_expanded.csv"))) {
            headers = Arrays.stream(br.readLine().split(",")).toList();
            String line;

            while ((line = br.readLine()) != null) {
                String[] segments = line.split(",");
                Target target = new Target();

                for (int i = 0; i < segments.length - 1; i++) {
                    target.when(headers.get(i), Double.parseDouble(segments[i]));
                }

                target.targetIs(Double.parseDouble(segments[segments.length - 1]));
                training.add(target);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }

        ArrayList<Target> testing = new ArrayList<>();
        for(int i = 0; i < Math.floor(training.size()*0.2); i++) {
            testing.add(training.get(i));
            training.remove(i);
        }

        System.out.println("Training data size: " + training.size());
        System.out.println("Testing data size: " + testing.size());

        ArrayList<String> headersCopy = new ArrayList<>(headers);
        headersCopy.remove(headersCopy.size() - 1);
        VarNode.variables = Arrays.asList(headersCopy.toArray(new String[0]));

        SymbolicRegressionProblem sgpTrainingData = new SymbolicRegressionProblem(baseFunctionNodeTypes, baseTerminalNodeTypes, training);
        SymbolicRegressionProblem sgpTestingData = new SymbolicRegressionProblem(baseFunctionNodeTypes, baseTerminalNodeTypes, testing);

        Task<ProgramSolution, ProgramProblem> symbolicRegressionTask = new Task<>(sgpTrainingData, StopCriterion.EVALUATIONS, 100000, 0, 0);
        GPAlgorithm alg = new ElitismGPAlgorithm();

        try {
            ProgramSolution solution = alg.execute(symbolicRegressionTask);
            System.out.println("Fitness on Training Data -> " + solution.getEval());

            solution.getTree().displayTree("stem_cell_equasion", true);
            System.out.println(solution);

            sgpTestingData.evaluate(solution);
            System.out.println("Fitness on Testing Data -> " + solution.getEval());

        } catch (StopCriterionException e) {
            e.printStackTrace();
        }
        //https://www.kaggle.com/datasets/ricardoluis/stem-cell-content-prediction-in-bioprocess

    }
}
