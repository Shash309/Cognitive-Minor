import React, { useEffect, useState, useMemo } from "react";
import './CollegeExplorer.css';
import { useTranslation } from 'react-i18next';

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:5000/api";

const CollegeCard = ({ college, animationDelay, isTier1 }) => {
  const { t } = useTranslation();
  const getSafeWebsite = (url) => {
    if (!url || url.trim() === "") return null;
    return url.startsWith("http://") || url.startsWith("https://")
      ? url
      : `https://${url}`;
  };

  const websiteUrl = getSafeWebsite(college.website);

  return (
    <div
      className={`college-card ${isTier1 ? 'tier-1-card' : ''}`}
      style={{ animationDelay: `${animationDelay}ms` }}
    >
      {isTier1 && <div className="featured-tag">{t('college.featured')}</div>}
      <div className="card-content">
        <h3>{college.college_name}</h3>
        <p><i className="fas fa-map-marker-alt"></i> {college.city}</p>
      </div>

      {college.rankings?.length > 0 && (
        <div className="rankings-container">
          {college.rankings.map((ranking, i) => (
            <div key={i} className="ranking-item">
              <span className="ranking-field">{ranking.field}</span>
              <span className="ranking-score">
                {t('college.rank')}: <b>{ranking.ranking ?? "N/A"}</b> | {t('college.score')}: <b>{ranking.score ?? "N/A"}</b>
              </span>
            </div>
          ))}
        </div>
      )}

      {websiteUrl && (
        <a href={websiteUrl} target="_blank" rel="noopener noreferrer" className="btn-visit">
          {t('common.visitWebsite')} <i className="fas fa-external-link-alt"></i>
        </a>
      )}
    </div>
  );
};

const CollegeExplorer = () => {
  const { t } = useTranslation();
  const [states, setStates] = useState([]);
  const [selectedState, setSelectedState] = useState("");
  const [colleges, setColleges] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchStates = async () => {
      try {
        const res = await fetch(`${API_BASE}/states`);
        if (!res.ok) throw new Error(`HTTP error! ${res.status}`);
        const data = await res.json();
        setStates(data.sort());
      } catch (err) {
        console.error("❌ Error fetching states:", err.message);
        setError(t('college.errorStates'));
      }
    };
    fetchStates();
  }, [t]);

  const handleStateChange = (state) => {
    setSelectedState(state);
    fetchColleges(state);
  };

  const fetchColleges = async (state) => {
    if (!state) return;
    setLoading(true);
    setError("");
    setColleges([]);
    try {
      const res = await fetch(`${API_BASE}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `HTTP error! ${res.status}`);
      }

      const data = await res.json();
      if (!Array.isArray(data)) throw new Error("Invalid API response");

      setColleges(data);
    } catch (err) {
      console.error("❌ Error fetching colleges:", err.message);
      setError(`${t('college.errorCollegesPrefix')}${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  /* Debugging: Log colleges when they change */
  useEffect(() => {
    if (colleges.length > 0) {
      console.log("✅ Colleges loaded:", colleges.length, colleges[0]);
    }
  }, [colleges]);

  const getBestRanking = (college) => {
    if (!college.rankings?.length) return Infinity;
    const valid = college.rankings
      .map((r) => r.ranking)
      .filter((r) => r !== null && r !== undefined);
    return valid.length ? Math.min(...valid) : Infinity;
  };

  const tier1 = colleges.filter((c) => getBestRanking(c) <= 10);
  const tier2 = colleges.filter((c) => {
    const r = getBestRanking(c);
    return r > 10 && r <= 25;
  });
  // Fix: Include unranked colleges (Infinity) in 'Others'
  const others = colleges.filter(
    (c) => {
      const r = getBestRanking(c);
      return r > 25 || r === Infinity;
    }
  );

  return (
    <div className="college-explorer-container">
      <div className="college-explorer-header">
        <h2 className="explorer-title">{t('college.title')}</h2>
        <select
          className="state-dropdown"
          value={selectedState}
          onChange={(e) => handleStateChange(e.target.value)}
        >
          <option value="" disabled>{t('college.selectStatePlaceholder')}</option>
          {states.map((st, i) => (
            <option key={i} value={st}>{st}</option>
          ))}
        </select>
      </div>

      {loading && <div className="loading-spinner"><div></div><div></div><div></div></div>}

      {error && (
        <div className="error-container">
          <p className="error-message">{error}</p>
          <button className="btn-retry" onClick={() => fetchColleges(selectedState)}>
            <i className="fas fa-redo"></i> {t('common.retry')}
          </button>
        </div>
      )}

      {!loading && !error && selectedState && (
        <div className="results-container">
          {tier1.length > 0 && (
            <div className="college-section">
              <h3>{t('college.tier1InState', { state: selectedState })}</h3>
              <div className="college-list">
                {tier1.map((c, i) => <CollegeCard key={c.college_name + i} college={c} animationDelay={i * 100} isTier1={true} />)}
              </div>
            </div>
          )}
          {tier2.length > 0 && (
            <div className="college-section">
              <h3>{t('college.tier2InState', { state: selectedState })}</h3>
              <div className="college-list">
                {tier2.map((c, i) => <CollegeCard key={c.college_name + i} college={c} animationDelay={(tier1.length + i) * 100} />)}
              </div>
            </div>
          )}
          {others.length > 0 && (
            <div className="college-section">
              <h3>{t('college.othersInState', { state: selectedState })}</h3>
              <div className="college-list">
                {others.map((c, i) => <CollegeCard key={c.college_name + i} college={c} animationDelay={(tier1.length + tier2.length + i) * 100} />)}
              </div>
            </div>
          )}
          {colleges.length === 0 && (
            <div className="no-colleges-message">
              <p>{t('college.noCollegesForState', { state: selectedState })}</p>
            </div>
          )}
        </div>
      )}

      {!loading && !error && !selectedState && (
        <div className="no-colleges-message">
          <p>{t('college.promptSelectState')}</p>
        </div>
      )}
    </div>
  );
};

export default CollegeExplorer;