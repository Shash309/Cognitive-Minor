import React, { useState, useEffect, useMemo, useCallback } from 'react';
import './TimelineTracker.css';
import { useTranslation } from 'react-i18next';

// --- REAL-TIME DATA (as of late 2024 for the 2025 academic year) ---
// This array is now populated with the most current, real-world dates.
// --- REAL-TIME DATA (Updated for 2026 Academic Year) ---
const realTimeEventsData = [
  {
    title: 'JEE Main 2026 (Session 2) Registration',
    date: '2026-03-02T21:00:00',
    category: 'Entrance Exam',
    link: 'https://jeemain.nta.ac.in/'
  },
  {
    title: 'NEET UG 2026 Application',
    date: '2026-03-09T23:50:00',
    category: 'Entrance Exam',
    link: 'https://exams.nta.ac.in/NEET/'
  },
  {
    title: 'UPSC CSE (Prelims) 2026 Application',
    date: '2026-03-05T18:00:00',
    category: 'Government Exam',
    link: 'https://upsc.gov.in/'
  },
  {
    title: 'VITEEE 2026 Application Deadline',
    date: '2026-03-30T23:59:59',
    category: 'Entrance Exam',
    link: 'https://viteee.vit.ac.in/'
  },
  {
    title: 'CUET UG 2026 Registration',
    date: '2026-03-31T21:00:00',
    category: 'Entrance Exam',
    link: 'https://exams.nta.ac.in/CUET-UG/'
  },
  {
    title: 'BITSAT 2026 Application',
    date: '2026-04-10T17:00:00',
    category: 'Entrance Exam',
    link: 'https://www.bitsadmission.com/'
  },
  {
    title: 'CAT 2026 Registration (Expected)',
    date: '2026-08-02T10:00:00',
    category: 'Entrance Exam',
    link: 'https://iimcat.ac.in/'
  },
  {
    title: 'GATE 2027 Registration Starts',
    date: '2026-08-30T10:00:00',
    category: 'Entrance Exam',
    link: 'https://gate.iitk.ac.in/'
  },
  {
    title: 'National Scholarship Portal 2026-27',
    date: '2026-10-31T23:59:59',
    category: 'Scholarship',
    link: 'https://scholarships.gov.in/'
  }
];

// --- Helper Components ---
const getEventStatus = (date, t) => {
  const now = new Date();
  const eventDate = new Date(date);
  const thirtyDaysFromNow = new Date(now.getTime() + 30 * 24 * 60 * 60 * 1000);
  if (eventDate < now) return { text: t('timeline.status.closed'), priority: 3 };
  if (eventDate <= thirtyDaysFromNow) return { text: t('timeline.status.active'), priority: 1 };
  return { text: t('timeline.status.upcoming'), priority: 2 };
};

const TimeLeft = ({ date }) => {
  const { i18n } = useTranslation();
  const calculateTimeLeft = useCallback(() => {
    const difference = +new Date(date) - +new Date();
    if (difference <= 0) return {};
    return {
      days: Math.floor(difference / (1000 * 60 * 60 * 24)),
      hours: Math.floor((difference / (1000 * 60 * 60)) % 24),
      minutes: Math.floor((difference / 1000 / 60) % 60),
      seconds: Math.floor((difference / 1000) % 60),
    };
  }, [date]);

  const [timeLeft, setTimeLeft] = useState(calculateTimeLeft());

  useEffect(() => {
    const timer = setInterval(() => setTimeLeft(calculateTimeLeft()), 1000);
    return () => clearInterval(timer);
  }, [calculateTimeLeft]);

  if (!Object.keys(timeLeft).length) return null;

  return (
    <div className="countdown">
      {Object.entries(timeLeft).map(([interval, value]) => (
        <div key={interval} className="countdown-item">
          <div className="countdown-value">{value}</div>
          <div className="countdown-label">{interval}</div>
        </div>
      ))}
    </div>
  );
};

// --- Main TimelineTracker Component ---
const TimelineTracker = () => {
  const { t, i18n } = useTranslation();
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState('All');
  const [searchTerm, setSearchTerm] = useState('');

  const fetchEvents = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      await new Promise(resolve => setTimeout(resolve, 1000));

      const dataWithStatus = realTimeEventsData.map(event => ({
        ...event,
        status: getEventStatus(event.date, t),
      }));
      setEvents(dataWithStatus);

    } catch (err) {
      setError(t('timeline.errorFetch'));
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => {
    fetchEvents();
  }, [fetchEvents]);

  const filteredAndSortedEvents = useMemo(() => {
    return events
      .filter(event => {
        const matchesFilter = filter === 'All' || event.category === filter;
        const matchesSearch = event.title.toLowerCase().includes(searchTerm.toLowerCase());
        return matchesFilter && matchesSearch;
      })
      .sort((a, b) => a.status.priority - b.status.priority || new Date(a.date) - new Date(b.date));
  }, [events, filter, searchTerm]);

  return (
    <div className="timeline-container">
      <div className="timeline-header">
        <h2>{t('timeline.title')}</h2>
        <p>{t('timeline.subtitle')}</p>
      </div>

      <div className="timeline-controls">
        <div className="filter-buttons">
          <button className={filter === 'All' ? 'active' : ''} onClick={() => setFilter('All')}>{t('timeline.filters.all')}</button>
          <button className={filter === 'Entrance Exam' ? 'active' : ''} onClick={() => setFilter('Entrance Exam')}>{t('timeline.filters.exams')}</button>
          <button className={filter === 'Scholarship' ? 'active' : ''} onClick={() => setFilter('Scholarship')}>{t('timeline.filters.scholarships')}</button>
          <button className={filter === 'Admission' ? 'active' : ''} onClick={() => setFilter('Admission')}>{t('timeline.filters.admissions')}</button>
        </div>
        <div className="search-box-container">
          <i className="fas fa-search"></i>
          <input
            type="text"
            className="search-box"
            placeholder={t('timeline.searchPlaceholder')}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>

      <div className="timeline">
        {loading && (
          <div className="loading-spinner"><div></div><div></div><div></div></div>
        )}
        {error && (
          <div className="error-container">
            <p className="error-message">{error}</p>
            <button className="btn-retry" onClick={fetchEvents}>
              <i className="fas fa-redo"></i> {t('common.retry')}
            </button>
          </div>
        )}
        {!loading && !error && (
          filteredAndSortedEvents.length > 0 ? (
            filteredAndSortedEvents.map((event, index) => (
              <div key={index} className={`timeline-item ${index % 2 === 0 ? 'left' : 'right'}`} style={{ animationDelay: `${index * 100}ms` }}>
                <div className="timeline-dot"></div>
                <div className="event-card">
                  <div className={`status-tag ${event.status.text.toLowerCase()}`}>{event.status.text}</div>
                  <div className="event-details">
                    <span className={`category-chip ${event.category.toLowerCase().replace(' ', '-')}`}>{event.category}</span>
                    <h3>{event.title}</h3>
                    <p><i className="fas fa-calendar-day"></i> {t('timeline.deadlineLabel')} {new Date(event.date).toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' })}</p>
                  </div>
                  <div className="event-actions">
                    <TimeLeft date={event.date} />
                    <a href={event.link} target="_blank" rel="noopener noreferrer" className="btn-apply">
                      {t('common.viewDetails')} <i className="fas fa-external-link-alt"></i>
                    </a>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="no-events-message">
              <p>{t('timeline.noEvents')}</p>
            </div>
          )
        )}
      </div>
    </div>
  );
};

export default TimelineTracker;