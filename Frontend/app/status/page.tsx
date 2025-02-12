import { CronMonitor } from "@/components/cron-monitor"
import { LogViewer } from "@/components/log-viewer"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ActivitySquare } from "lucide-react"
import prisma from "@/lib/prisma"

export const dynamic = "force-dynamic"
export const revalidate = 60 // Revalidate every minute

async function getConsultations() {
  try {
    const consultations = await prisma.consultation.findMany({
      orderBy: {
        createdAt: 'desc'
      },
      take: 10, // Last 10 consultations
    });
    return consultations;
  } catch (error) {
    console.error('Failed to fetch consultations:', error);
    return [];
  }
}

export default async function StatusPage() {
  const consultations = await getConsultations();

  return (
    <div className="container py-8 max-w-5xl mx-auto space-y-8">
      <Card className="border-none shadow-none bg-transparent">
        <CardHeader className="text-center space-y-4">
          <div className="mx-auto bg-primary/10 w-fit p-3 rounded-full">
            <ActivitySquare className="w-8 h-8 text-primary" />
          </div>
          <div className="space-y-2">
            <CardTitle className="text-4xl font-bold">System Status</CardTitle>
            <CardDescription className="text-lg">
              Monitor system health and view application logs
            </CardDescription>
          </div>
        </CardHeader>
      </Card>

      <Card className="border shadow-md">
        <CardHeader>
          <CardTitle>Recent Consultations</CardTitle>
          <CardDescription>View recent patient consultation history</CardDescription>
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
                          Consultation ID: {consultation.id}
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

                    <Alert>
                      <AlertDescription>
                        <div className="font-medium mb-2">Patient Information</div>
                        {Object.entries(consultation.initialResponses).map(([key, value]) => (
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
                          {consultation.followupResponses.map((response, index) => (
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

      <Card className="border shadow-md">
        <CardHeader>
          <CardTitle>Cron Job Status</CardTitle>
          <CardDescription>Monitor the health of scheduled tasks</CardDescription>
        </CardHeader>
        <CardContent>
          <CronMonitor />
        </CardContent>
      </Card>

      <Card className="border shadow-md">
        <CardHeader>
          <CardTitle>System Logs</CardTitle>
          <CardDescription>View detailed application logs</CardDescription>
        </CardHeader>
        <CardContent>
          <LogViewer />
        </CardContent>
      </Card>
    </div>
  )
}