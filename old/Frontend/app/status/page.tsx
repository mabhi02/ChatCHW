// app/status/page.tsx
'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ActivitySquare, Users, Clock, MessageSquare, Languages, Volume2 } from "lucide-react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts'

interface ConsultationData {
  id: string
  createdAt: Date
  status: string
  language: string
  initialResponses: any
  followupResponses: any[]
  analytics?: {
    totalDuration: number
    messageCount: number
    languageChanges: number
    voiceInteractions: number
    completionStatus: string
  }
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

export default function StatusPage() {
  const [consultations, setConsultations] = useState<ConsultationData[]>([]);
  const [analytics, setAnalytics] = useState({
    totalConsultations: 0,
    activeConsultations: 0,
    averageDuration: 0,
    completionRate: 0,
    languageDistribution: {} as Record<string, number>,
    consultationTrend: [] as { date: string; count: number }[]
  });

  const fetchData = async () => {
    try {
      const response = await fetch('/api/consultations');
      const data = await response.json();
      setConsultations(data.consultations);
      setAnalytics(data.analytics);
    } catch (error) {
      console.error('Failed to fetch data:', error);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="container py-8 max-w-7xl mx-auto space-y-8">
      <Card className="border-none shadow-none bg-transparent">
        <CardHeader className="text-center space-y-4">
          <div className="mx-auto bg-primary/10 w-fit p-3 rounded-full">
            <ActivitySquare className="w-8 h-8 text-primary" />
          </div>
          <div className="space-y-2">
            <CardTitle className="text-4xl font-bold">System Status</CardTitle>
            <CardDescription className="text-lg">
              Real-time monitoring and analytics
            </CardDescription>
          </div>
        </CardHeader>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">Total Consultations</p>
                <p className="text-2xl font-bold">{analytics.totalConsultations}</p>
              </div>
              <Users className="w-8 h-8 text-primary/60" />
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">Active Sessions</p>
                <p className="text-2xl font-bold">{analytics.activeConsultations}</p>
              </div>
              <Clock className="w-8 h-8 text-primary/60" />
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">Avg. Duration</p>
                <p className="text-2xl font-bold">{Math.round(analytics.averageDuration / 60)}m</p>
              </div>
              <MessageSquare className="w-8 h-8 text-primary/60" />
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">Completion Rate</p>
                <p className="text-2xl font-bold">{Math.round(analytics.completionRate * 100)}%</p>
              </div>
              <Languages className="w-8 h-8 text-primary/60" />
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <Card className="border shadow-md">
          <CardHeader>
            <CardTitle>Consultation Trend</CardTitle>
            <CardDescription>Number of consultations over time</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={analytics.consultationTrend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="count" stroke="#8884d8" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card className="border shadow-md">
          <CardHeader>
            <CardTitle>Language Distribution</CardTitle>
            <CardDescription>Usage across different languages</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={Object.entries(analytics.languageDistribution).map(([name, value]) => ({
                      name,
                      value
                    }))}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    fill="#8884d8"
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {Object.entries(analytics.languageDistribution).map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="border shadow-md">
        <CardHeader>
          <CardTitle>Recent Consultations</CardTitle>
          <CardDescription>View recent consultation history and analytics</CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[400px] pr-4">
            <div className="space-y-4">
              {consultations.map((consultation) => (
                <Card key={consultation.id} className="p-4">
                  <div className="space-y-2">
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="font-medium">
                          ID: {consultation.id}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {new Date(consultation.createdAt).toLocaleString()}
                        </div>
                      </div>
                      <div className={`px-2 py-1 rounded-full text-xs ${
                        consultation.status === 'COMPLETED' 
                          ? 'bg-green-500/10 text-green-500' 
                          : consultation.status === 'IN_PROGRESS'
                          ? 'bg-blue-500/10 text-blue-500'
                          : 'bg-red-500/10 text-red-500'
                      }`}>
                        {consultation.status}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 py-2">
                      <div className="text-sm">
                        <span className="text-muted-foreground">Messages:</span>
                        <br />
                        {consultation.analytics?.messageCount || 0}
                      </div>
                      <div className="text-sm">
                        <span className="text-muted-foreground">Duration:</span>
                        <br />
                        {Math.round((consultation.analytics?.totalDuration || 0) / 60)}m
                      </div>
                      <div className="text-sm">
                        <span className="text-muted-foreground">Language Changes:</span>
                        <br />
                        {consultation.analytics?.languageChanges || 0}
                      </div>
                      <div className="text-sm">
                        <span className="text-muted-foreground">Voice Interactions:</span>
                        <br />
                        {consultation.analytics?.voiceInteractions || 0}
                      </div>
                    </div>

                    <Alert>
                      <AlertDescription>
                        <div className="font-medium mb-2">Initial Responses</div>
                        {Object.entries(consultation.initialResponses).map(([key, value]: [string, any]) => (
                          <div key={key} className="text-sm">
                            {value.question}: {
                              Array.isArray(value.answer) 
                                ? value.answer.join(", ") 
                                : value.answer
                            }
                          </div>
                        ))}
                      </AlertDescription>
                    </Alert>

                    {consultation.followupResponses && consultation.followupResponses.length > 0 && (
                      <div className="mt-4">
                        <div className="font-medium mb-2">Follow-up Responses</div>
                        <div className="space-y-2">
                          {consultation.followupResponses.map((response: any, index: number) => (
                            <div key={index} className="text-sm">
                              Q: {response.question}
                              <br />
                              A: {response.answer}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </Card>
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  )
}