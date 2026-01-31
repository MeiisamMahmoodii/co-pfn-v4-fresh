import React, { useState, useEffect } from 'react';

// --- Styled Components / Design Tokens ---
const colors = {
  bg: '#0a0a0c',
  card: '#16161a',
  primary: '#7c3aed',
  primaryLight: '#a78bfa',
  secondary: '#3b82f6',
  accent: '#10b981',
  text: '#f1f1f2',
  textDim: '#a1a1aa',
  border: '#27272a',
  codeBg: '#1e1e24'
};

const Card = ({ children, title, subtitle }) => (
  <div style={{
    backgroundColor: colors.card,
    borderRadius: '16px',
    padding: '24px',
    border: `1px solid ${colors.border}`,
    boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
    height: '100%',
    display: 'flex',
    flexDirection: 'column'
  }}>
    {title && <h3 style={{ margin: '0 0 4px 0', color: colors.primaryLight, fontSize: '1.2rem' }}>{title}</h3>}
    {subtitle && <p style={{ margin: '0 0 16px 0', color: colors.textDim, fontSize: '0.9rem' }}>{subtitle}</p>}
    <div style={{ flex: 1 }}>{children}</div>
  </div>
);

const CodeBlock = ({ code, language = 'python' }) => (
  <pre style={{
    backgroundColor: colors.codeBg,
    padding: '16px',
    borderRadius: '8px',
    fontSize: '0.85rem',
    color: '#d4d4d8',
    overflowX: 'auto',
    border: `1px solid ${colors.border}`,
    lineHeight: '1.5'
  }}>
    <code>{code}</code>
  </pre>
);

const Badge = ({ children, color = colors.primary }) => (
  <span style={{
    backgroundColor: `${color}22`,
    color: color,
    padding: '4px 10px',
    borderRadius: '999px',
    fontSize: '0.75rem',
    fontWeight: '600',
    border: `1px solid ${color}44`,
    marginLeft: '8px'
  }}>
    {children}
  </span>
);

// --- Main Components ---

const Hero = () => (
  <div style={{ textAlign: 'center', padding: '60px 20px', background: `linear-gradient(135deg, ${colors.bg} 0%, #1a1a2e 100%)` }}>
    <h1 style={{ fontSize: '3.5rem', margin: '0', fontWeight: '800', background: `linear-gradient(to right, ${colors.primaryLight}, ${colors.secondary})`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
      Co-PFN: Constraint-aware PFN
    </h1>
    <p style={{ fontSize: '1.4rem', color: colors.textDim, marginTop: '10px' }}>
      Adversarial Trust Auditing for Causal Inference
    </p>
    <div style={{ marginTop: '20px' }}>
      <Badge color={colors.accent}>Phase 6: Soft Alignment</Badge>
      <Badge color={colors.secondary}>Supervisor-Ready Report</Badge>
    </div>
  </div>
);

const Objective = () => (
  <section style={{ padding: '40px 20px' }}>
    <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
      <Card title="The Core Objective" subtitle="Moving from Blind Trust to Empirical Auditing">
        <p style={{ color: colors.text, lineHeight: '1.6', fontSize: '1.1rem' }}>
          Standard causal inference methods often rely on untestable assumptions. <strong>Co-PFN</strong> introduces an
          <em>"Evidence-First"</em> auditing system that verifies expert causal claims against observational data.
          The model decides whether to <strong>Trust</strong> a claim based on structural signatures in the data,
          using high-trust signals to refine Average Treatment Effect (ATE) estimates.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px', marginTop: '20px' }}>
          <div style={{ padding: '15px', borderRadius: '12px', background: '#ffffff05', border: '1px solid #ffffff10' }}>
            <h4 style={{ color: colors.secondary, marginBottom: '8px' }}>The Problem</h4>
            <p style={{ fontSize: '0.9rem', margin: 0 }}>If a causal claim is wrong, traditional estimates are biased. Experts can be mistaken or biased.</p>
          </div>
          <div style={{ padding: '15px', borderRadius: '12px', background: '#00ff0005', border: '1px solid #00ff0010' }}>
            <h4 style={{ color: colors.accent, marginBottom: '8px' }}>Our Solution</h4>
            <p style={{ fontSize: '0.9rem', margin: 0 }}>Model treats claims as queries and data as evidence, gating the correction of ATE through a learnable trust score.</p>
          </div>
        </div>
      </Card>
    </div>
  </section>
);

const MetricsDashboard = () => (
  <section style={{ padding: '40px 20px' }}>
    <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
      <h2 style={{ textAlign: 'center', marginBottom: '30px' }}>Live Audit Metrics</h2>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
        <Card title="Corruption Sensitivity" subtitle="Discrimination between True vs False claims">
          <div style={{ textAlign: 'center', padding: '10px' }}>
            <div style={{ fontSize: '2.5rem', fontWeight: '700', color: colors.accent }}>0.7360</div>
            <div style={{ fontSize: '0.8rem', color: colors.textDim }}>True Claim Mean Trust</div>
            <div style={{ height: '40px' }} />
            <div style={{ fontSize: '2.5rem', fontWeight: '700', color: '#ef4444' }}>0.1195</div>
            <div style={{ fontSize: '0.8rem', color: colors.textDim }}>False Claim Mean Trust</div>
            <div style={{ marginTop: '20px', padding: '10px', backgroundColor: '#ffffff05', borderRadius: '8px' }}>
              Gap: <span style={{ fontWeight: 'bold', color: colors.primaryLight }}>0.6165</span> (Strong Separation)
            </div>
          </div>
        </Card>

        <Card title="Real-World Transfer" subtitle="Lalonde (1986) Dataset Validation">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '10px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>Valid Demographic Adj.</span>
              <span style={{ fontWeight: '600', color: colors.accent }}>0.5976</span>
            </div>
            <div style={{ height: '8px', background: '#333', borderRadius: '4px' }}>
              <div style={{ width: '59.7%', height: '100%', background: colors.accent, borderRadius: '4px' }} />
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '10px' }}>
              <span>Reverse Causality (Y→T)</span>
              <span style={{ fontWeight: '600', color: '#ef4444' }}>0.0712</span>
            </div>
            <div style={{ height: '8px', background: '#333', borderRadius: '4px' }}>
              <div style={{ width: '7.1%', height: '100%', background: '#ef4444', borderRadius: '4px' }} />
            </div>
          </div>
        </Card>

        <Card title="Garbage Rejection" subtitle="Null Dataset Audit (Gaussian Noise)">
          <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <div style={{ position: 'relative', width: '150px', height: '150px' }}>
              <svg viewBox="0 0 100 100" style={{ transform: 'rotate(-90deg)' }}>
                <circle cx="50" cy="50" r="45" fill="none" stroke="#222" strokeWidth="8" />
                <circle cx="50" cy="50" r="45" fill="none" stroke={colors.primary} strokeWidth="8" strokeDasharray="282.7" strokeDashoffset="0.1" />
              </svg>
              <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center' }}>
                <div style={{ fontWeight: '700', fontSize: '1.5rem' }}>99.9%</div>
                <div style={{ fontSize: '0.6rem', color: colors.textDim }}>REJECTION</div>
              </div>
            </div>
            <p style={{ fontSize: '0.9rem', marginTop: '15px', textAlign: 'center' }}>Model successfully ignores claims when data has no causal signal.</p>
          </div>
        </Card>
      </div>
    </div>
  </section>
);

const ArchitectureDiagram = () => (
  <section style={{ padding: '40px 20px', backgroundColor: '#0c0c0e' }}>
    <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
      <h2 style={{ textAlign: 'center', marginBottom: '40px' }}>Theory-First Transformer Architecture</h2>
      <div style={{
        padding: '40px',
        backgroundColor: colors.card,
        borderRadius: '20px',
        border: `1px solid ${colors.border}`,
        position: 'relative'
      }}>
        <svg viewBox="0 0 800 500" style={{ width: '100%', height: 'auto' }}>
          <rect x="50" y="50" width="150" height="60" rx="8" fill="#3b82f633" stroke={colors.secondary} strokeWidth="2" />
          <text x="125" y="85" textAnchor="middle" fill="#fff" fontSize="14">Observational Data</text>

          <rect x="50" y="160" width="150" height="60" rx="8" fill="#7c3aed33" stroke={colors.primary} strokeWidth="2" />
          <text x="125" y="195" textAnchor="middle" fill="#fff" fontSize="14">TabPFN Encoder</text>

          <rect x="50" y="270" width="150" height="60" rx="8" fill="#7c3aed33" stroke={colors.primary} strokeWidth="2" />
          <text x="125" y="305" textAnchor="middle" fill="#fff" fontSize="14">Base Transformer</text>

          <rect x="300" y="380" width="200" height="60" rx="8" fill="#10b98133" stroke={colors.accent} strokeWidth="2" />
          <text x="400" y="415" textAnchor="middle" fill="#fff" fontSize="14">Final ATE Output</text>

          <rect x="550" y="50" width="150" height="60" rx="8" fill="#7c3aed33" stroke={colors.primary} strokeWidth="2" />
          <text x="625" y="85" textAnchor="middle" fill="#fff" fontSize="14">Causal Claims</text>

          <rect x="550" y="160" width="150" height="60" rx="8" fill="#7c3aed33" stroke={colors.primary} strokeWidth="2" />
          <text x="625" y="195" textAnchor="middle" fill="#fff" fontSize="14">Theory Transformer</text>

          <rect x="325" y="220" width="150" height="80" rx="8" fill="#f59e0b33" stroke="#f59e0b" strokeWidth="2" />
          <text x="400" y="260" textAnchor="middle" fill="#fff" fontSize="14">Cross-Attention</text>
          <text x="400" y="280" textAnchor="middle" fill="#fff" fontSize="10">(Adversarial Auditor)</text>

          <rect x="350" y="320" width="100" height="40" rx="8" fill="#ffffff11" stroke="#ffffff33" strokeWidth="2" />
          <text x="400" y="345" textAnchor="middle" fill="#fff" fontSize="12">Correction Gate</text>

          <path d="M125 110 L125 160" stroke="#444" strokeWidth="2" fill="none" markerEnd="url(#arrow)" />
          <path d="M125 220 L125 270" stroke="#444" strokeWidth="2" fill="none" markerEnd="url(#arrow)" />
          <path d="M125 330 L125 410 L300 410" stroke="#444" strokeWidth="2" fill="none" markerEnd="url(#arrow)" />

          <path d="M625 110 L625 160" stroke="#444" strokeWidth="2" fill="none" markerEnd="url(#arrow)" />
          <path d="M625 220 L625 260 L475 260" stroke="#444" strokeWidth="2" fill="none" markerEnd="url(#arrow)" />
          <path d="M200 300 L325 260" stroke="#444" strokeWidth="2" fill="none" markerEnd="url(#arrow)" />

          <path d="M400 300 L400 320" stroke="#444" strokeWidth="2" fill="none" markerEnd="url(#arrow)" />
          <path d="M400 360 L400 380" stroke="#444" strokeWidth="2" fill="none" markerEnd="url(#arrow)" />

          <defs>
            <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,6 L9,3 z" fill="#444" />
            </marker>
          </defs>
        </svg>
      </div>
    </div>
  </section>
);

const TechnicalDetails = () => (
  <section style={{ padding: '40px 20px' }}>
    <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
      <h2 style={{ textAlign: 'center', marginBottom: '30px' }}>Technical Deep-Dive</h2>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '30px' }}>
        <Card title="Trust Amplification" subtitle="Forcing Decisive Sigmoids">
          <p style={{ fontSize: '0.9rem', marginBottom: '15px' }}>
            To overcome neural network conservatism (clustering at 0.5), we apply a learnable temperature scale
            before the activation.
          </p>
          <CodeBlock code={`# src/models/core.py
trust_amplified = torch.sigmoid(
    trust_raw * torch.relu(self.trust_scale)
)`} />
        </Card>

        <Card title="Gated Correction" subtitle="How Claims Impact the ATE">
          <p style={{ fontSize: '0.9rem', marginBottom: '15px' }}>
            The final estimate is a sum of the data-only baseline and the theory-based correction,
            multiplied by the amplified trust gate.
          </p>
          <CodeBlock code={`# The "Evidence-Only" Flow
theory_evidence, _ = self.cross_attn(Q=claims, K=data, V=data)
trust_scores = torch.sigmoid(self.trust_head(theory_evidence))

# Detached Trust prevents gradient cheating
gated_theory = theory_evidence * trust_scores.detach()
correction = self.correction_head(gated_theory)

# Result
final_ate = base_ate + correction`} />
        </Card>
      </div>
    </div>
  </section>
);

const InitiativesRoadmap = () => {
  const initiatives = [
    { title: "Ground-Truth Verification", desc: "Using networkx d-separation to ensure 'True' labels are mathematically sound.", status: "Done" },
    { title: "Hard Negative Mining", desc: "Generating 'Trap' claims like reverse causation and mediators to force structural learning.", status: "Done" },
    { title: "Pairwise Ranking Loss", desc: "Optimizing the GAP between True/False claims rather than absolute scores.", status: "Done" },
    { title: "Trust Amplification", desc: "Sharpening decision boundaries for real-world transfer.", status: "Done" },
    { title: "Soft Alignment (Ph 6)", desc: "Calibrating trust scores with ATE estimation utility to avoid decoupling.", status: "Next" }
  ];

  return (
    <section style={{ padding: '40px 20px', backgroundColor: '#0c0c0e' }}>
      <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
        <h2 style={{ textAlign: 'center', marginBottom: '30px' }}>Project Initiatives & Roadmap</h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
          {initiatives.map((init, i) => (
            <div key={i} style={{
              display: 'flex',
              alignItems: 'center',
              padding: '20px',
              backgroundColor: colors.card,
              borderRadius: '12px',
              borderLeft: `5px solid ${init.status === 'Done' ? colors.accent : init.status === 'In Progress' ? colors.secondary : colors.primary}`
            }}>
              <div style={{ flex: 1 }}>
                <h4 style={{ margin: 0, color: colors.text }}>{init.title}</h4>
                <p style={{ margin: '5px 0 0 0', fontSize: '0.9rem', color: colors.textDim }}>{init.desc}</p>
              </div>
              <Badge color={init.status === 'Done' ? colors.accent : init.status === 'In Progress' ? colors.secondary : colors.primary}>
                {init.status}
              </Badge>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

const Footer = () => (
  <footer style={{ textAlign: 'center', padding: '40px', color: colors.textDim, fontSize: '0.8rem', borderTop: `1px solid ${colors.border}` }}>
    <p>© 2026 Co-PFN Research Group | Adversarial Causal Auditing Project</p>
    <p>Generated for Audit Report v4.2</p>
  </footer>
);

export default function ReportInfographic() {
  return (
    <div style={{
      backgroundColor: colors.bg,
      color: colors.text,
      minHeight: '100vh',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
    }}>
      <Hero />
      <Objective />
      <InitiativesRoadmap />
      <ArchitectureDiagram />
      <MetricsDashboard />
      <TechnicalDetails />
      <Footer />
    </div>
  );
}