import React, { useState, useCallback, useEffect } from 'react';
import ReactFlow, { MiniMap, Controls, Background, useNodesState, useEdgesState, MarkerType } from 'reactflow';
import { getLayoutedElements } from './layout.js';
import 'reactflow/dist/style.css';
import './CareerPathVisualizer.css';
import { useTranslation } from 'react-i18next';

// --- Custom Modern Node ---
const IconNode = ({ data }) => {
  return (
    <div className={`mindmap-node ${data.category} ${data.isPath ? 'active-path' : ''}`}>
      {data.icon && <i className={`node-icon ${data.icon}`}></i>}
      <div className="node-label">{data.label}</div>
    </div>
  );
};

const nodeTypes = { mindmap: IconNode, input: IconNode, output: IconNode, default: IconNode };

const fullDataset = {
  nodes: [
    // Level 0: School
    { id: '10th', type: 'mindmap', data: { label: 'Class 10th', icon: 'fas fa-school', category: 'school', level: 0, description: 'Secondary Education.' } },

    // Level 1: Higher Secondary
    { id: '12-sci-pcm', type: 'mindmap', data: { label: '12th Science (PCM)', icon: 'fas fa-atom', category: 'high-school', level: 1, description: 'Focus on Maths & Physics.' } },
    { id: '12-sci-pcb', type: 'mindmap', data: { label: '12th Science (PCB)', icon: 'fas fa-dna', category: 'high-school', level: 1, description: 'Focus on Biology & Chemistry.' } },
    { id: '12-comm', type: 'mindmap', data: { label: '12th Commerce', icon: 'fas fa-chart-bar', category: 'high-school', level: 1, description: 'Focus on Business & Accounts.' } },
    { id: '12-arts', type: 'mindmap', data: { label: '12th Arts', icon: 'fas fa-paint-brush', category: 'high-school', level: 1, description: 'Focus on Humanities.' } },

    // Level 2: Undergrad
    { id: 'ug-btech-cse', type: 'mindmap', data: { label: 'B.Tech CSE', icon: 'fas fa-code', category: 'college', level: 2, description: 'Tech & Software.' } },
    { id: 'ug-btech-mech', type: 'mindmap', data: { label: 'B.Tech Mech', icon: 'fas fa-cogs', category: 'college', level: 2, description: 'Mechanical Systems.' } },
    { id: 'ug-mbbs', type: 'mindmap', data: { label: 'MBBS', icon: 'fas fa-user-md', category: 'college', level: 2, description: 'Medical Science.' } },
    { id: 'ug-bsc-bio', type: 'mindmap', data: { label: 'B.Sc Biology', icon: 'fas fa-microscope', category: 'college', level: 2, description: 'Life Sciences.' } },
    { id: 'ug-bcom', type: 'mindmap', data: { label: 'B.Com / BBA', icon: 'fas fa-briefcase', category: 'college', level: 2, description: 'Business Studies.' } },
    { id: 'ug-ba-psych', type: 'mindmap', data: { label: 'BA Psychology', icon: 'fas fa-brain', category: 'college', level: 2, description: 'Psychology.' } },
    { id: 'ug-ba-polsci', type: 'mindmap', data: { label: 'BA Pol. Science', icon: 'fas fa-globe', category: 'college', level: 2, description: 'Political Science.' } },

    // Level 3: Postgrad
    { id: 'pg-mtech', type: 'mindmap', data: { label: 'M.Tech', icon: 'fas fa-microchip', category: 'postgrad', level: 3, description: 'Engineering Specialization.' } },
    { id: 'pg-mba', type: 'mindmap', data: { label: 'MBA', icon: 'fas fa-user-tie', category: 'postgrad', level: 3, description: 'Business Management.' } },
    { id: 'pg-md', type: 'mindmap', data: { label: 'MD / MS', icon: 'fas fa-hospital', category: 'postgrad', level: 3, description: 'Medical Specialization.' } },
    { id: 'pg-ma', type: 'mindmap', data: { label: 'MA / MSc', icon: 'fas fa-book-open', category: 'postgrad', level: 3, description: 'Advanced Studies.' } },

    // Level 4: Careers
    { id: 'job-swe', type: 'mindmap', data: { label: 'Software Engineer', icon: 'fas fa-laptop-code', category: 'career', level: 4, description: 'Build software systems.' } },
    { id: 'job-datasci', type: 'mindmap', data: { label: 'Data Scientist', icon: 'fas fa-database', category: 'career', level: 4, description: 'Data analysis & AI.' } },
    { id: 'job-mech-eng', type: 'mindmap', data: { label: 'Mech. Engineer', icon: 'fas fa-tools', category: 'career', level: 4, description: 'Industrial design.' } },
    { id: 'job-doctor', type: 'mindmap', data: { label: 'Doctor', icon: 'fas fa-stethoscope', category: 'career', level: 4, description: 'Healthcare professional.' } },
    { id: 'job-researcher', type: 'mindmap', data: { label: 'Researcher', icon: 'fas fa-flask', category: 'career', level: 4, description: 'Scientific exploration.' } },
    { id: 'job-banker', type: 'mindmap', data: { label: 'Investment Banker', icon: 'fas fa-money-bill-wave', category: 'career', level: 4, description: 'Finance expert.' } },
    { id: 'job-manager', type: 'mindmap', data: { label: 'Product Manager', icon: 'fas fa-tasks', category: 'career', level: 4, description: 'Product strategy.' } },
    { id: 'job-civil', type: 'mindmap', data: { label: 'Civil Servant', icon: 'fas fa-university', category: 'career', level: 4, description: 'Public Administration.' } },
    { id: 'job-psych', type: 'mindmap', data: { label: 'Psychologist', icon: 'fas fa-head-side-virus', category: 'career', level: 4, description: 'Consulting Expert.' } },
  ],
  edges: [
    { id: 'e1', source: '10th', target: '12-sci-pcm' },
    { id: 'e2', source: '10th', target: '12-sci-pcb' },
    { id: 'e3', source: '10th', target: '12-comm' },
    { id: 'e4', source: '10th', target: '12-arts' },
    { id: 'e5', source: '12-sci-pcm', target: 'ug-btech-cse' },
    { id: 'e6', source: '12-sci-pcm', target: 'ug-btech-mech' },
    { id: 'e7', source: '12-sci-pcm', target: 'ug-bcom' },
    { id: 'e8', source: '12-sci-pcb', target: 'ug-mbbs' },
    { id: 'e9', source: '12-sci-pcb', target: 'ug-bsc-bio' },
    { id: 'e10', source: '12-comm', target: 'ug-bcom' },
    { id: 'e11', source: '12-arts', target: 'ug-ba-psych' },
    { id: 'e12', source: '12-arts', target: 'ug-ba-polsci' },
    { id: 'e13', source: 'ug-btech-cse', target: 'pg-mtech' },
    { id: 'e14', source: 'ug-btech-cse', target: 'pg-mba' },
    { id: 'e15', source: 'ug-btech-mech', target: 'pg-mba' },
    { id: 'e16', source: 'ug-mbbs', target: 'pg-md' },
    { id: 'e17', source: 'ug-bsc-bio', target: 'pg-ma' },
    { id: 'e18', source: 'ug-bcom', target: 'pg-mba' },
    { id: 'e19', source: 'ug-ba-psych', target: 'pg-ma' },
    { id: 'e20', source: 'ug-ba-polsci', target: 'pg-ma' },
    { id: 'e21', source: 'ug-btech-cse', target: 'job-swe' },
    { id: 'e22', source: 'ug-btech-mech', target: 'job-mech-eng' },
    { id: 'e23', source: 'ug-bcom', target: 'job-banker' },
    { id: 'e24', source: 'pg-mtech', target: 'job-datasci' },
    { id: 'e25', source: 'pg-mtech', target: 'job-swe' },
    { id: 'e26', source: 'pg-mba', target: 'job-manager' },
    { id: 'e27', source: 'pg-mba', target: 'job-banker' },
    { id: 'e28', source: 'pg-md', target: 'job-doctor' },
    { id: 'e29', source: 'pg-ma', target: 'job-researcher' },
    { id: 'e30', source: 'pg-ma', target: 'job-psych' },
    { id: 'e31', source: 'pg-ma', target: 'job-civil' },
    { id: 'e32', source: 'ug-ba-polsci', target: 'job-civil' },
  ]
};

// --- Panel Content ---
const SidePanel = ({ node, onClose }) => {
  if (!node) return null;
  return (
    <div className={`side-panel ${node ? 'open' : ''}`}>
      <button className="close-button" onClick={onClose}><i className="fas fa-times"></i></button>
      <div className="panel-content">
        <div className="panel-header">
          <div className={`panel-indicator ${node.data.category}`}></div>
          <h3>{node.data.label}</h3>
        </div>
        <p>{node.data.description || "Learn more about this academic or career stage."}</p>
        <div className="panel-stats">
          <div className="stat">Stage: <span>{node.data.category}</span></div>
        </div>
      </div>
    </div>
  );
};

const PathTimeline = ({ steps }) => {
  if (!steps || steps.length === 0) return null;

  return (
    <div className="roadmap-timeline">
      <h3><i className="fas fa-route"></i> Career Roadmap</h3>
      <div className="timeline-steps">
        {steps.map((step, index) => (
          <div key={step.id} className="timeline-step">
            <div className={`step-circle ${step.data.category}`}>
              <i className={step.data.icon}></i>
            </div>
            <div className="step-info">
              <span className="step-label">{step.data.label}</span>
              <span className="step-category">{step.data.category.replace('-', ' ')}</span>
            </div>
            {index < steps.length - 1 && <div className="step-connector"><i className="fas fa-chevron-right"></i></div>}
          </div>
        ))}
      </div>
    </div>
  );
};

const CareerPathVisualizer = () => {
  const { t } = useTranslation();
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [orderedPath, setOrderedPath] = useState([]);

  useEffect(() => {
    let pathNodeIds = new Set();
    let pathEdgeIds = new Set();

    if (selectedNode) {
      pathNodeIds.add(selectedNode.id);
      const findAncestors = (nodeId) => {
        fullDataset.edges.filter(e => e.target === nodeId).forEach(e => {
          pathEdgeIds.add(e.id);
          pathNodeIds.add(e.source);
          findAncestors(e.source);
        });
      };
      findAncestors(selectedNode.id);

      const findDescendants = (nodeId) => {
        fullDataset.edges.filter(e => e.source === nodeId).forEach(e => {
          pathEdgeIds.add(e.id);
          pathNodeIds.add(e.target);
          findDescendants(e.target);
        });
      };
      findDescendants(selectedNode.id);

      // Create ordered path for the timeline
      const pathArray = fullDataset.nodes
        .filter(n => pathNodeIds.has(n.id))
        .sort((a, b) => a.data.level - b.data.level);
      setOrderedPath(pathArray);
    } else {
      setOrderedPath([]);
    }

    const currentNodes = fullDataset.nodes.map(n => ({
      ...n,
      data: { ...n.data, isPath: pathNodeIds.has(n.id) },
      style: { opacity: selectedNode && !pathNodeIds.has(n.id) ? 0.3 : 1 }
    }));

    const currentEdges = fullDataset.edges.map(e => ({
      ...e,
      type: 'smoothstep',
      animated: pathEdgeIds.has(e.id),
      style: {
        stroke: pathEdgeIds.has(e.id) ? '#3b82f6' : '#475569',
        strokeWidth: pathEdgeIds.has(e.id) ? 4 : 1,
        strokeDasharray: pathEdgeIds.has(e.id) ? '10 5' : 'none', // Dotted line effect
        opacity: selectedNode && !pathEdgeIds.has(e.id) ? 0.1 : 1,
      },
      zIndex: pathEdgeIds.has(e.id) ? 999 : 1,
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: pathEdgeIds.has(e.id) ? '#3b82f6' : '#475569',
      },
    }));

    getLayoutedElements(currentNodes, currentEdges).then(({ nodes: layoutedNodes }) => {
      setNodes(layoutedNodes);
      setEdges(currentEdges);
    });

  }, [selectedNode, setNodes, setEdges]);

  const onNodeClick = useCallback((event, node) => setSelectedNode(node), []);
  const onPaneClick = useCallback(() => setSelectedNode(null), []);

  return (
    <div className="visualizer-container">
      <div className="visualizer-header">
        <h2>{t('visualizer.title', 'Career Journey Visualizer')}</h2>
        <p>Interactive path mapping from Class 10th to your target Career. Select any stage to trace your route.</p>
      </div>
      <div className="visualizer-wrapper">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={onNodeClick}
          onPaneClick={onPaneClick}
          nodeTypes={nodeTypes}
          fitView
          proOptions={{ hideAttribution: true }}
        >
          <Controls />
          <Background variant="dots" gap={40} size={1} color="rgba(255,255,255,0.05)" />
        </ReactFlow>
        <SidePanel node={selectedNode} onClose={() => setSelectedNode(null)} />
      </div>

      <PathTimeline steps={orderedPath} />
    </div>
  );
};

export default CareerPathVisualizer;
