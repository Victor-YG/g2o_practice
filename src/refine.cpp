#include <random>
#include <iostream>

#include <Eigen/Core>

#include "g2o/types/slam3d/edge_pointxyz.h"
#include "g2o/types/slam3d/vertex_pointxyz.h"
#include "g2o/core/block_solver.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/core/optimization_algorithm_levenberg.h"


int main(int argc, char** argv)
{
    // problem specification
    const int grid_size = 10;
    const int max_iterations = 100;
    const double noise_amp_vert = 0.5;
    const double noise_amp_edge = 0.01;
    std::default_random_engine generator;
    std::normal_distribution<double> dist_vert(0, noise_amp_vert);
    std::normal_distribution<double> dist_edge(0, noise_amp_edge);
    Eigen::Matrix<double, 3, 3> information = Eigen::Matrix<double, 3, 3>::Identity();

    // setup optimization
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(true);
    
    g2o::OptimizationAlgorithmLevenberg* solver =
        new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolverX>(
            g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>()));

    optimizer.setAlgorithm(solver);

    // add vertices
    std::vector<std::vector<g2o::VertexPointXYZ*>> vertices;

    vertices.resize(grid_size);
    for (int i = 0; i < grid_size; i++)
    {
        vertices[i].resize(grid_size);
    }

    int idx = 0;
    for (int r = 0; r < grid_size; r++)
    {
        for (int c = 0; c < grid_size; c++)
        {
            g2o::VertexPointXYZ* v = new g2o::VertexPointXYZ();
            v->setId(idx++);
            double x_err = dist_vert(generator);
            double y_err = dist_vert(generator);
            Eigen::Vector3d p(c + x_err, r + y_err, 0.0);
            std::cout << "[INFO]: Initial position of vertex<" 
            << r << ", " << c << ">: = "
            << "(" << p(0) << ", " << p(1) << ", " << p(2) << ")" << std::endl;
            v->setEstimate(p);
            vertices[r][c] = v;
            optimizer.addVertex(v);
        }
    }

    // fix vertex (0, 0)
    vertices[0][0]->setEstimate(Eigen::Vector3d(0, 0, 0));
    vertices[0][0]->setFixed(true);

    // add edges
    std::vector<g2o::EdgePointXYZ*> edges;

    // create horizontal edges
    for (int r = 0; r < grid_size; r++)
    {
        for (int i = 0; i < grid_size - 1; i++)
        {
            g2o::VertexPointXYZ* prev = vertices[r][i];
            g2o::VertexPointXYZ* curr = vertices[r][i + 1];
            double x_err = dist_edge(generator);
            double y_err = dist_edge(generator);
            Eigen::Vector3d t(1.0 + x_err, 0.0 + y_err, 0.0);
            g2o::EdgePointXYZ* e = new g2o::EdgePointXYZ();
            e->setVertex(0, prev);
            e->setVertex(1, curr);
            e->setMeasurement(t);
            e->setInformation(information);
            edges.emplace_back(e);
            optimizer.addEdge(e);
        }
    }

    // create vertical edges
    for (int c = 0; c < grid_size; c++)
    {
        for (int i = 0; i < grid_size - 1; i++)
        {
            g2o::VertexPointXYZ* prev = vertices[i][c];
            g2o::VertexPointXYZ* curr = vertices[i + 1][c];
            double x_err = dist_edge(generator);
            double y_err = dist_edge(generator);
            Eigen::Vector3d t(0.0 + x_err, 1.0 + y_err, 0.0);
            g2o::EdgePointXYZ* e = new g2o::EdgePointXYZ();
            e->setVertex(0, prev);
            e->setVertex(1, curr);
            e->setMeasurement(t);
            e->setInformation(information);
            edges.emplace_back(e);
            optimizer.addEdge(e);
        }
    }

    // optimize
    optimizer.initializeOptimization();
    optimizer.optimize(max_iterations);
    std::cout << "[INFO]: Done optimization." << std::endl;

    // post analysis
    for (int r = 0; r < grid_size; r++)
    {
        for (int c = 0; c < grid_size; c++)
        {
            g2o::VertexPointXYZ* v = vertices[r][c];
            Eigen::Vector3d p = v->estimate();
            std::cout << "[INFO]: Refined position of vertex<" 
            << r << ", " << c << ">: = "
            << "(" << p(0) << ", " << p(1) << ", " << p(2) << ")" << std::endl;
        }
    }

    std::cout << "[INFO]: End of program." << std::endl;
    return 0;
}
