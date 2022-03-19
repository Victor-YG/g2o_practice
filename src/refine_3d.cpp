#include <fstream>
#include <sstream>

#include <Eigen/Core>

#include "gflags/gflags.h"

#include "g2o/types/slam3d/edge_se3.h"
#include "g2o/types/slam3d/vertex_se3.h"

#include "g2o/core/block_solver.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/core/optimization_algorithm_levenberg.h"


DEFINE_string(input, "./../res/pose_graph.txt", 
    "Input file containing vertices (x y z qx qy qz qw) and edges (src_id dst_id x y z qx qy qz qw).");
DEFINE_int32(max_iter, 10, "Maximum refinement interation.");


void load_problem(
    const std::string& filename, 
    g2o::SparseOptimizer& optimizer, 
    std::vector<g2o::VertexSE3*>& vertices, 
    std::vector<g2o::EdgeSE3*>& edges);

void save_as_ply(
    const std::string& filename, 
    std::vector<g2o::VertexSE3*>& vertices, 
    std::vector<g2o::EdgeSE3*>& edges);


int main(int argc, char** argv)
{
    // setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(true);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolverX>(
            g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>()
        )
    );

    optimizer.setAlgorithm(solver);

    // load problem
    std::vector<g2o::VertexSE3*> vertices;
    std::vector<g2o::EdgeSE3*> edges;
    // std::string filename = "./pose_graph.txt";
    load_problem(FLAGS_input, optimizer, vertices, edges);

    save_as_ply("./original.ply", vertices, edges);

    // optimize
    auto v = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertices().find(0)->second);
    v->setFixed(true);
    bool has_gauge_freedom = optimizer.gaugeFreedom();

    optimizer.initializeOptimization();
    optimizer.optimize(FLAGS_max_iter);
    std::cout << "[INFO]: Done optimization." << std::endl;

    save_as_ply("./refined.ply", vertices, edges);

    return 0;
}

void load_problem(
    const std::string& filename, 
    g2o::SparseOptimizer& optimizer, 
    std::vector<g2o::VertexSE3*>& vertices, 
    std::vector<g2o::EdgeSE3*>& edges)
{
    std::ifstream input;
    input.open(filename);
    if (!input.is_open()) return;

    // read header
    std::string header;
    std::getline(input, header);
    int n_vertices;
    int n_edges;

    std::stringstream ss(header);
    ss >> n_vertices >> n_edges;

    std::cout << "num_of_vertices = " << n_vertices << std::endl;
    std::cout << "num_of_edges = " << n_edges << std::endl;

    vertices.clear();
    edges.clear();
    vertices.reserve(n_vertices);
    edges.reserve(n_edges);

    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
    information(0, 0) = 0.01;
    information(1, 1) = 0.01;
    information(2, 2) = 0.01;

    // read vertices
    for (int i = 0; i < n_vertices; i++)
    {
        std::string line;
        std::getline(input, line);
        
        double x, y, z, qx, qy, qz, qw;

        std::stringstream ss(line);
        ss >> x >> y >> z >> qx >> qy >> qz >> qw;
        std::array<double, 7> qt = { x, y, z, qx, qy, qz, qw };

        g2o::VertexSE3* v = new g2o::VertexSE3();
        v->setId(i);
        v->setEstimateData(&qt[0]);
        optimizer.addVertex(v);
        vertices.emplace_back(v);
    }

    // read edges
    for (int i = 0; i < n_edges; i++)
    {
        std::string line;
        std::getline(input, line);
        
        int a, b;
        double x, y, z, qx, qy, qz, qw;

        std::stringstream ss(line);
        ss >> a >> b >> x >> y >> z >> qx >> qy >> qz >> qw;
        std::array<double, 7> qt = { x, y, z, qx, qy, qz, qw };

        g2o::EdgeSE3* e = new g2o::EdgeSE3();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
            optimizer.vertices().find(a)->second)
        );
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
            optimizer.vertices().find(b)->second)
        );
        e->setMeasurementData(&qt[0]);
        e->setInformation(information);
        optimizer.addEdge(e);
        edges.emplace_back(e);
    }
}

void save_as_ply(
    const std::string& filename, 
    std::vector<g2o::VertexSE3*>& vertices, 
    std::vector<g2o::EdgeSE3*>& edges)
{
    std::ofstream output;
    output.open(filename);

    if (!output.is_open()) return;
    int N = vertices.size(), M = edges.size();
    
    // write header
    output << "ply" << std::endl;
    output << "format ascii 1.0" << std::endl;
    output << "comment object: list of points" << std::endl;
    output << "element vertex " << N << std::endl;
    output << "property float x" << std::endl;
    output << "property float y" << std::endl;
    output << "property float z" << std::endl;
    output << "element edge " << M << std::endl;
    output << "property int vertex1" << std::endl;
    output << "property int vertex2" << std::endl;
    output << "end_header" << std::endl;

    // write vertices
    for (int i = 0; i < N; i++)
    {
        g2o::VertexSE3* v = vertices[i];
        std::array<double, 7> qt;
        v->getEstimateData(&qt[0]);
        output << qt[0] << " " << qt[1] << " " << qt[2] << std::endl;
    }

    // write edges
    for (int i = 0; i < M; i++)
    {
        g2o::EdgeSE3* e = edges[i];
        int a = e->vertex(0)->id();
        int b = e->vertex(1)->id();
        output << a << " " << b << std::endl;
    }
}